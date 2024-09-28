# Optimized Speculative Sampling for GPU Hardware Accelerators

This is the code for the paper [Optimized Speculative Sampling for GPU Hardware Accelerators](https://arxiv.org/abs/2406.11016). 

## Run 

**Requirements:** Python >= 3.9 and the requirements listed in `requirements.txt` (Install with `pip install -r requirements.txt`). 

Language modeling tasks are executed with `run_llm.py` and ASR tasks are executed with `run_asr.py`. 

The most important command line flags:

- `--with_profiling`: Enables the torch profiler.  
- `--use_custom_sampler`: Enables the custom CUDA kernel for sampling (if the argument isn't set, the default implementation provided by Huggingface will be used)
- `--do_eval`: Computes performance metrics such as WER and ROUGE score

### LLMs 

A typical LLM inference run looks as follows:

```shell
dataset_name="cnn_dailymail"
subset_name="3.0.0"
text_column="article"
reference_text_column="highlights"
max_new_tokens=128

# Run with exact kernel:
# kernel_name="speculative_sampling_fp32"

# Run with sigmoid sampling kernel:
kernel_name="speculative_sampling_fp32_sigmoid"

model_name="google/gemma-7b"
assistant_model_name="google/gemma-2b"
python run_llm.py \
    --model_name "${model_name}" \
    --assistant_model_name "${assistant_model_name}" \
    --with_profiling \
    --use_custom_sampler \
    --kernel_name "${kernel_name}" \
    --max_new_tokens ${max_new_tokens} \
    --dataset "${dataset_name}" \
    --subset "${subset_name}" \
    --text_column "${text_column}" \
    --reference_text_column "${reference_text_column}" \
    --do_eval \
    --output_suffix "${host}_${max_new_tokens}T_"
```


### Whisper 

A typical ASR run looks as follows:

```shell
model_name="openai/whisper-small.en"
assistant_model_name="distil-whisper/distil-small.en"
dataset="librispeech_asr"
subset="clean"

# Run with sigmoid kernel:
# kernel_name="speculative_sampling_half_sigmoid"

# Run with exact kernel:
kernel_name="speculative_sampling_half"

python run_asr.py \
  --dataset ${dataset} \
  --subset ${subset} \
  --with_profiling \
  --use_custom_sampler \
  --kernel_name ${kernel_name} \
  --model_name ${model_name} \
  --assistant_model_name ${assistant_model_name} \
  --output_suffix "ASR_" || exit 1
```

# Debugging with cuda-gdb

## Compilation 

Set the `-g` and `-G` flags as extra options to `nvcc`.
The `-g` option enables debugging symbols, while `-G` retains the GPU debugging information.

Example: 

```python
custom_sampling_module = load(name='custom_sampling', sources=['main.cpp', f"{cli_args.kernel_name}.cu"], extra_cuda_cflags=['-O2', '-g', `-G`])
```

## Debugger 

```shell
export PATH=$PATH:/usr/local/cuda-12.3/bin
cuda-gdb -q --args python run_asr.py --with_profiling --use_custom_sampler --kernel_name speculative_sampling_half

(cuda-gdb) set cuda break_on_launch application
(cuda-gdb) run
```

## Memcheck

```shell
compute-sanitizer python run_asr.py --use_custom_sampler --kernel_name speculative_sampling_half
```

You should get outputs like this in case there are illegal memory accesses:
```shell
========= Invalid __global__ read of size 2 bytes
=========     at division_kernel(c10::Half *, c10::Half *, c10::Half *, c10::Half *, c10::Half *, int, int, int, int)+0x4a0
=========     by thread (0,0,0) in block (687,0,0)
=========     Address 0x7fdc7b40015e is out of bounds
=========     and is 351 bytes after the nearest allocation at 0x7fdc7b200000 of size 2097152 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x3344e0]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========     Host Frame: [0x1498c]
=========                in /data/user/wagnerdo/libtorch-test/self-speculative-decoding/venv/lib/python3.9/site-packages/torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12
=========     Host Frame:cudaLaunchKernel [0x6bedb]
=========                in /data/user/wagnerdo/libtorch-test/self-speculative-decoding/venv/lib/python3.9/site-packages/torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12
=========     Host Frame:__device_stub__Z15division_kernelPN3c104HalfES1_S1_S1_S1_iiii(c10::Half*, c10::Half*, c10::Half*, c10::Half*, c10::Half*, int, int, int, int) in /tmp/tmpxft_001c5df6_00000000-6_speculative_hf_half_reorg.cudafe1.stub.c:14 [0x79b34]
=========                in /home/wagnerdo/.cache/torch_extensions/py39_cu121/custom_sampling/custom_sampling.so
=========     Host Frame:sampling_cuda(at::Tensor, at::Tensor, int, at::Tensor, bool, int) in /data/user/wagnerdo/libtorch-test/self-speculative-decoding/speculative_hf_half_reorg.cu:143 [0x7a178]
=========                in /home/wagnerdo/.cache/torch_extensions/py39_cu121/custom_sampling/custom_sampling.so
=========     Host Frame:std::enable_if<!std::is_member_pointer<std::decay<std::tuple<at::Tensor, int> (* const&)(at::Tensor, at::Tensor, int, at::Tensor, bool, int)>::type>::value, std::invoke_result<std::tuple<at::Tensor, int> (* const&)(at::Tensor, at::Tensor, int, at::Tensor, bool, int), at::Tensor, at::Tensor, int, at::Tensor, bool, int>::type>::type c10::guts::invoke<std::tuple<at::Tensor, int> (* const&)(at::Tensor, at::Tensor, int, at::Tensor, bool, int), at::Tensor, at::Tensor, int, at::Tensor, bool, int>(std::tuple<at::Tensor, int> (* const&)(at::Tensor, at::Tensor, int, at::Tensor, bool, int), at::Tensor&&, at::Tensor&&, int&&, at::Tensor&&, bool&&, int&&) [0x6092b]
=========                in /home/wagnerdo/.cache/torch_extensions/py39_cu121/custom_sampling/custom_sampling.so
```

# Citation 

If you use this work in your research, please cite:

```bibtex
@inproceedings{
    wagner2024optimized,
    title={Optimized Speculative Sampling for {GPU} Hardware Accelerators},
    author={Dominik Wagner, Seanie Lee, Ilja Baumann, Philipp Seeberger, Korbinian Riedhammer, Tobias Bocklet},
    booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
    year={2024}
}
```