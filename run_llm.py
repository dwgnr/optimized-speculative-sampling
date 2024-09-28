import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from tqdm import tqdm
from transformers import set_seed
from custom_model.model_wrapper import (
    CustomLlamaForCausalLM,
    CustomQwen2ForCausalLM,
    CustomGemmaForCausalLM,
)
from evaluate import load as eval_load

set_seed(424242)


def get_model_cls(model_name: str):
    if "llama" in model_name:
        return CustomLlamaForCausalLM
    elif "Qwen" in model_name:
        return CustomQwen2ForCausalLM
    elif "gemma" in model_name:
        return CustomGemmaForCausalLM
    else:
        raise ValueError(f"Given model name '{model_name}' is not supported!")


def main(cli_args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    assert cli_args.num_passes > 0

    custom_sampling_fn = None
    if cli_args.use_custom_sampler:
        try:
            from torch.utils.cpp_extension import load

            custom_sampling_module = load(
                name="custom_sampling",
                sources=["main.cpp", f"{cli_args.kernel_name}.cu"],
                extra_cuda_cflags=["-O2"],
            )
            logging.info(
                f"Built custom sampling extension {custom_sampling_module} {type(custom_sampling_module)}"
            )
            custom_sampling_fn = custom_sampling_module.sampling_cuda
        except Exception as e:
            logging.error(e)

    logging.info(f"{device=}, {torch_dtype=}, {custom_sampling_fn=}")

    if not cli_args.run_hf_default_model:
        model = get_model_cls(cli_args.model_name).from_pretrained(
            cli_args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        logging.info(f"Running custom model class {model.__class__.__name__}.")

    else:
        logging.info("Running HF default model class.")
        if cli_args.use_custom_sampler:
            raise ValueError(
                "--use_custom_sampler is only available when --run_hf_default_model is not set!"
            )
        model = AutoModelForCausalLM.from_pretrained(
            cli_args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        )

    model.generation_config.return_dict_in_generate = True
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cli_args.model_name)

    assistant_model = AutoModelForCausalLM.from_pretrained(
        cli_args.assistant_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    )

    assistant_model.generation_config.num_assistant_tokens = (
        cli_args.num_assistant_tokens
    )  # default = 5
    logging.info(f"{assistant_model.generation_config.num_assistant_tokens=}")

    assistant_model.to(device)
    logging.info(
        f"Starting with {model.__class__.__name__} model ({model.dtype=}) and {assistant_model.__class__.__name__} assistant model ({assistant_model.dtype=})"
    )

    def assisted_generate_with_time(model, inputs, max_new_tokens=128, **kwargs):
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            assistant_model=assistant_model,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        generation_time = time.time() - start_time
        return outputs, generation_time

    logging.info(f"Loading dataset {cli_args.dataset}")

    dataset = load_dataset(
        cli_args.dataset, cli_args.subset, split=cli_args.split, trust_remote_code=True
    )
    if cli_args.max_samples is not None and cli_args.max_samples > 0:
        if cli_args.skip_shuffle:
            logging.info(
                f"Skipping shuffling and taking first {cli_args.max_samples} samples."
            )
            dataset = dataset.select(range(min(cli_args.max_samples, len(dataset))))
        else:
            logging.info(
                f"Taking {cli_args.max_samples} random samples from the dataset"
            )
            dataset = dataset.shuffle(seed=424242).select(
                range(min(cli_args.max_samples, len(dataset)))
            )

    all_time = 0
    predictions = []
    references = []
    profiling_results = []
    profiling_tables = []
    for sample in tqdm(dataset):
        if isinstance(sample, dict):
            t = sample[cli_args.text_column]
            if "dailymail" in cli_args.dataset or "xsum" in cli_args.dataset:
                t += " Summary:"
        else:
            t = sample

        if cli_args.is_chat_model:
            t = f"{cli_args.prompt} {t}"
            t = tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(t, return_tensors="pt").to(device=device)

        custom_model_kwargs = {}
        if not cli_args.run_hf_default_model:
            custom_model_kwargs = {
                "with_profiling": cli_args.with_profiling,
                "custom_speculative_sampler": custom_sampling_fn,
            }
        try:
            k_outputs = []
            k_passes = []
            for k in range(cli_args.num_passes):
                logging.info(f"Doing pass {k}/{cli_args.num_passes}, {len(k_passes)=}")
                output, gen_time = assisted_generate_with_time(
                    model,
                    inputs,
                    max_new_tokens=cli_args.max_new_tokens,
                    **custom_model_kwargs,
                )
                all_time += gen_time
                pred = tokenizer.batch_decode(
                    output.sequences, skip_special_tokens=True
                )[0]
                if cli_args.is_chat_model:
                    pred = pred.split(cli_args.prompt)[1].strip()
                if "dailymail" in cli_args.dataset or "xsum" in cli_args.dataset:
                    k_passes = pred.split("Summary:")[1].strip()
                elif "humaneval" in cli_args.dataset:
                    k_passes.append(pred)
                k_outputs.append(output)
                del output
                torch.cuda.empty_cache()
            predictions.append(k_passes)
            if cli_args.do_eval and isinstance(sample, dict):
                references.append(sample[cli_args.reference_text_column])
            for output in k_outputs:
                profiling_results.append(output.profiling_results)
                profiling_tables.append(output.profiling_table)
        except Exception as e:
            logging.error(f"{custom_sampling_fn=} ({cli_args.dataset}): {e}")

    peak_mem = 0
    try:
        peak_mem = torch.cuda.max_memory_allocated()
    except Exception as e:
        logging.error(f"Unable to extract peak memory statistics! {e}")
    if cli_args.with_profiling:
        logging.info("Last profiling table:")
        logging.info(profiling_tables[-1][-1])
    logging.info(f"Predictions: {predictions[:3]}")
    logging.info(f"References: {references[:3]}")
    logging.info(f"Wall Time: {all_time}s")

    custom_kernel_name = (
        "hf_sampler" if custom_sampling_fn is None else f"{cli_args.kernel_name}"
    )
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target_model = cli_args.model_name.replace("/", "-").replace(".", "-")
    assistant_model = cli_args.assistant_model_name.replace("/", "-").replace(".", "-")

    profiler_path = os.path.join("profiler", "llm", cli_args.dataset.replace(".", "-"))
    Path(profiler_path).mkdir(exist_ok=True, parents=True)
    out_file = os.path.join(
        profiler_path,
        f"{custom_kernel_name}_tgt_{target_model}_ass_{assistant_model}_{cli_args.output_suffix}{time_str}.csv",
    )
    with open(out_file, "w") as tf:
        tf.write(
            "example,cuda_time_total,cpu_time_total,self_cuda_time_total,self_cpu_time_total,candidate_length,n_matches\n"
        )
        for ex, prof_results in enumerate(profiling_results):
            for line in prof_results:
                tf.write(f"{ex},{line}\n")

    eval_head_str = ""
    eval_str = ""
    if cli_args.do_eval:
        eval_head_str = ","
        eval_str = ","
        if "humaneval" in cli_args.dataset:
            try:
                # NOTE: It is recommended to use the official code to evaluate HumanEval:
                # https://github.com/openai/human-eval
                logging.info("Loading code_eval metric")
                # os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                code_eval = eval_load("code_eval")
                k_evals = [1, 2, 4, 5, 10, 100]
                k_evals = [i for i in k_evals if i <= cli_args.num_passes]
                logging.info(
                    f"Using k={k_evals}, {len(predictions)=}, {len(references)}"
                )
                all_passes_at_k = {}
                for k in k_evals:
                    # Take only the first k predictions for evaluation, otherwise we get wrong scores
                    input_preds = [elem[0:k] for elem in predictions]
                    pass_at_k, _ = code_eval.compute(
                        predictions=input_preds, references=references, k=[k]
                    )
                    all_passes_at_k = {**all_passes_at_k, **pass_at_k}
                logging.info(f"{all_passes_at_k=}")
                eval_head_str += ",".join([i for i in all_passes_at_k.keys()])
                eval_str += ",".join([str(i) for i in all_passes_at_k.values()])
            except Exception as e:
                logging.error(f"HumanEval failed {e}")
        else:
            logging.info("Loading Rouge metric")
            metrics = eval_load("rouge")
            rouge = metrics.compute(predictions=predictions, references=references)
            logging.info(f"ROUGE: {rouge}")
            eval_head_str += ",".join([i for i in rouge.keys()])
            eval_str += ",".join([str(i) for i in rouge.values()])

    out_file = os.path.join(
        profiler_path,
        f"wall_time_total_{custom_kernel_name}_tgt_{target_model}_ass_{assistant_model}_{cli_args.output_suffix}{time_str}.csv",
    )
    with open(out_file, "w") as tf:
        tf.write(f"wall_time{eval_head_str},peak_mem_bytes\n")
        tf.write(f"{all_time}{eval_str},{peak_mem}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_custom_sampler",
        action="store_true",
        help="Use custom cuda kernel for speculative sampling",
    )
    parser.add_argument(
        "--is_chat_model",
        action="store_true",
        help="Set this flag if you use an instruction-finetuned model",
    )
    parser.add_argument(
        "--run_hf_default_model",
        action="store_true",
        help="Use Huggingface default model class for experiment",
    )
    parser.add_argument(
        "--with_profiling",
        action="store_true",
        help="Use torch profiler to profile _speculative_sampling",
    )
    parser.add_argument(
        "--num_assistant_tokens",
        type=int,
        default=5,
        help="Number of tokens generated via the assistant model",
    )
    parser.add_argument(
        "--skip_shuffle",
        action="store_true",
        help="Don't shuffle",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--kernel_name",
        type=str,
        default="speculative_hf_half_less_mem_fp32",
        help="Number of tokens generated via the assistant model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="super_glue",
        help="Dataset i.e. the 'path' parameter in load_dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="copa",
        help="Subset i.e. the 'name' parameter in load_dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Train/test/validation split",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="premise",
        help="Dataset column that contains the input text.",
    )
    parser.add_argument(
        "--reference_text_column",
        type=str,
        default="highlights",
        help="Dataset column that contains the reference text.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt for chat models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Target model name",
    )
    parser.add_argument(
        "--assistant_model_name",
        type=str,
        default="princeton-nlp/Sheared-LLaMA-1.3B",
        help="Smaller assistant model name",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Name that can be appended to the output csv",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation on dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to take from the test set",
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        default=1,
        help="Number of passes through the dataset. Only relevant for HumanEval",
    )
    args = parser.parse_args()

    formatter = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >> %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    main(args)
