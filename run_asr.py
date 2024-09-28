import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm
from datasets import load_dataset, Audio
from custom_model.model_wrapper import CustomWhisperForConditionalGeneration
from transformers import set_seed
from evaluate import load as eval_load

set_seed(424242)


def main(cli_args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

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
            # dataset = dataset.select(range(min(cli_args.max_samples, len(dataset))))
            dataset = dataset.shuffle(seed=424242).select(
                range(min(cli_args.max_samples, len(dataset)))
            )

    if "common_voice" in cli_args.dataset:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if not cli_args.run_hf_default_model:
        logging.info("Running custom model class.")
        model = CustomWhisperForConditionalGeneration.from_pretrained(
            cli_args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
    else:
        logging.info("Running HF default model class.")
        if cli_args.use_custom_sampler:
            raise ValueError(
                "--use_custom_sampler is only available when --run_hf_default_model is not set!"
            )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cli_args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        )

    model.generation_config.return_dict_in_generate = True
    model.to(device)

    processor = AutoProcessor.from_pretrained(cli_args.model_name)

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

    model.config.forced_decoder_ids = None
    assistant_model.config.forced_decoder_ids = None

    def assisted_generate_wrapper(model, inputs, **kwargs):
        start_time = time.time()
        outputs = model.generate(
            **inputs, assistant_model=assistant_model, do_sample=True, **kwargs
        )
        generation_time = time.time() - start_time
        return outputs, generation_time

    all_time = 0
    predictions = []
    references = []
    profiling_results = []
    profiling_tables = []
    for sample in tqdm(dataset):
        try:
            audio = sample["audio"]
            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
            )
            inputs = inputs.to(device=device, dtype=torch_dtype)
            custom_model_kwargs = {}
            if not cli_args.run_hf_default_model:
                custom_model_kwargs = {
                    "with_profiling": cli_args.with_profiling,
                    "custom_speculative_sampler": custom_sampling_fn,
                }
            output, gen_time = assisted_generate_wrapper(
                model, inputs, **custom_model_kwargs
            )
            all_time += gen_time
            predictions.append(
                processor.batch_decode(
                    output.sequences, skip_special_tokens=True, normalize=True
                )[0]
            )
            references.append(
                processor.tokenizer._normalize(sample[cli_args.text_column])
            )
            profiling_results.append(output.profiling_results)
            profiling_tables.append(output.profiling_table)
        except Exception as e:
            logging.error(f"{custom_sampling_fn=} ({cli_args.dataset}): {e}")

    peak_mem = 0
    try:
        peak_mem = torch.cuda.max_memory_allocated()
    except Exception as e:
        logging.error(f"Unable to extract peak memory statistics! {e}")

    metrics = eval_load("wer")

    def clean_for_wer_metric(preds, refs):
        pred_results = []
        ref_results = []
        for p, r in zip(preds, refs):
            if len(p) < 1 or len(r) < 1:
                logging.warning(
                    f"Found empty string for prediction '{p}' and reference '{r}'!"
                )
                continue
            pred_results.append(p)
            ref_results.append(r)
        return pred_results, ref_results

    predictions_clean, references_clean = clean_for_wer_metric(predictions, references)
    wer = metrics.compute(predictions=predictions_clean, references=references_clean)
    logging.info(f"WER: {wer}")
    logging.info("Last profiling table:")
    logging.info(profiling_tables[-1][-1])
    logging.info(f"Predictions: {predictions[:3]}")
    logging.info(f"References: {references[:3]}")
    logging.info(f"Wall Time: {all_time}s")

    custom_kernel_name = (
        "hf_sampler" if custom_sampling_fn is None else f"{cli_args.kernel_name}"
    )
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    profiler_path = os.path.join(
        "profiler", "asr", cli_args.dataset.replace("/", "-"), cli_args.subset
    )
    target_model = cli_args.model_name.replace("/", "-").replace(".", "-")
    assistant_model = cli_args.assistant_model_name.replace("/", "-").replace(".", "-")
    Path(profiler_path).mkdir(exist_ok=True, parents=True)
    out_file = os.path.join(
        profiler_path,
        f"{custom_kernel_name}_tgt_{target_model}_ass_{assistant_model}_{cli_args.split}_{cli_args.output_suffix}{time_str}.csv",
    )
    with open(out_file, "w") as tf:
        tf.write(
            "example,cuda_time_total,cpu_time_total,self_cuda_time_total,self_cpu_time_total,candidate_length,n_matches\n"
        )
        for ex, prof_results in enumerate(profiling_results):
            for line in prof_results:
                tf.write(f"{ex},{line}\n")

    out_file = os.path.join(
        profiler_path,
        f"wall_time_total_{custom_kernel_name}_tgt_{target_model}_ass_{assistant_model}_{cli_args.split}_{cli_args.output_suffix}{time_str}.csv",
    )
    with open(out_file, "w") as tf:
        tf.write("wall_time,wer,peak_mem_bytes\n")
        tf.write(f"{all_time},{wer},{peak_mem}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_custom_sampler",
        action="store_true",
        help="Use custom cuda kernel for speculative sampling",
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
        "--skip_shuffle",
        action="store_true",
        help="Don't shuffle",
    )
    parser.add_argument(
        "--num_assistant_tokens",
        type=int,
        default=5,
        help="Number of tokens generated via the assistant model",
    )
    parser.add_argument(
        "--kernel_name",
        type=str,
        default="speculative_hf_half",
        help="Number of tokens generated via the assistant model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="Dataset i.e. the 'path' parameter in load_dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="clean",
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
        default="text",
        help="Dataset column that contains the references. Needed for WER computation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small.en",
        help="Target model name",
    )
    parser.add_argument(
        "--assistant_model_name",
        type=str,
        default="distil-whisper/distil-small.en",
        help="Smaller assistant model name",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Name that can be appended to the output csv",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to take from the test set",
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