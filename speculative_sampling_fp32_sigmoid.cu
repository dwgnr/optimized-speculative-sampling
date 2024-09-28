#include <iostream>
#include <torch/torch.h>
#include <cuda.h>
#include <cstdio>
#include <cuda_fp16.h>

#define MAX_LOGIT_VAL 10000
#define MIN_LOGIT_VAL -10000

using namespace torch::indexing;

__global__ void sampling_kernel_sigmoid(float* p, float* q, float* probability_ratio, float* p_prime, float* clamped_sum, int batch_size,
                                        int tokens, int vocab_size, int threads) {

    unsigned int ix = blockIdx.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * batch_size + ix;

    extern __shared__ float sram[];
    int tile_alloc = threads;
    float* p_tile = &sram[tile_alloc];
    float* q_tile = &sram[tile_alloc * 2];
    float* p_prime_tile = &sram[tile_alloc * 3];
    float* clamped_sum_tile = &sram[tile_alloc * 4];

    if (ix < batch_size && iy < tokens) {
        for (int tile_start = 0; tile_start < vocab_size; tile_start += threads) {
            int tile_end = min(tile_start + threads, vocab_size);
            int tile_size = tile_end - tile_start;
            unsigned int current_idx = threadIdx.x;

            if (current_idx < tile_size) {

                float elem_p = p[idx * vocab_size + tile_start + current_idx];
                float elem_q = q[idx * vocab_size + tile_start + current_idx];

                if (elem_p < MIN_LOGIT_VAL) {
                    p_tile[current_idx] = 0.0f;
                } else if (elem_p > MAX_LOGIT_VAL) {
                    p_tile[current_idx] = 1.0f;
                }
                else {
                    elem_p = (elem_p - MIN_LOGIT_VAL) / (MAX_LOGIT_VAL - MIN_LOGIT_VAL);
                    p_tile[current_idx] = elem_p >= 0 ? 1.0f / (1.0f + __expf(-elem_p)) : __expf(elem_p) / (1.0f + __expf(elem_p));
                }

                if (elem_q < MIN_LOGIT_VAL) {
                    q_tile[current_idx] = 0.0f;
                } else if (elem_q > MAX_LOGIT_VAL) {
                    q_tile[current_idx] = 1.0f;
                } else {
                    elem_q = (elem_q - MIN_LOGIT_VAL) / (MAX_LOGIT_VAL - MIN_LOGIT_VAL);
                    q_tile[current_idx] = elem_q >= 0 ? 1.0f / (1.0f + __expf(-elem_q)) : __expf(elem_q) / (1.0f + __expf(elem_q));
                }

                float tmp_diff = p_tile[current_idx] - q_tile[current_idx];
                tmp_diff = tmp_diff > 0.0f ? tmp_diff : 0.0f;

                clamped_sum_tile[current_idx] = tmp_diff;
                p_prime_tile[current_idx] = tmp_diff;

                __syncthreads();
                for (unsigned int s = tile_size / 2; s > 0; s >>= 1) {
                    if (current_idx < s) {
                        clamped_sum_tile[current_idx] += clamped_sum_tile[current_idx + s];
                    }
                    __syncthreads();
                }

                if (current_idx == 0) {
                    clamped_sum[idx] += clamped_sum_tile[0];
                }

                probability_ratio[idx * vocab_size + tile_start + current_idx] =  p_tile[current_idx] / q_tile[current_idx];
                p_prime[idx * vocab_size + tile_start + current_idx] = p_prime_tile[current_idx];
                p[idx * vocab_size + tile_start + current_idx] = p_tile[current_idx];
                __syncthreads();
            }
        }
    }
}

std::tuple<torch::Tensor, int>
sampling_cuda(torch::Tensor candidate_input_ids,
              torch::Tensor candidate_logits,
              const int candidate_length,
              torch::Tensor new_logits,
              const bool last_assistant_token_is_eos,
              const int max_matches) {

    TORCH_CHECK(new_logits.dim() == 3, "new_logits must be a 3D tensor");
    TORCH_CHECK(candidate_logits.dim() == 3, "candidate_logits must be a 3D tensor");
    TORCH_CHECK(candidate_input_ids.dim() == 2, "candidate_input_ids must be a 2D tensor");
    TORCH_CHECK(new_logits.size(0) == candidate_logits.size(0), "Batch size mismatch");
    TORCH_CHECK(new_logits.size(2) == candidate_logits.size(2), "Vocab size mismatch");
    TORCH_CHECK(candidate_logits.dtype() == torch::kFloat32, "candidate_logits tensor must be of type torch::kFloat32");
    TORCH_CHECK(new_logits.dtype() == torch::kFloat32, "new_logits tensor must be of type torch::kFloat32");

    const int batch_size = candidate_logits.size(0);
    const int tokens_new = new_logits.size(1);
    const int tokens_cand = candidate_logits.size(1);
    const int vocab_size = candidate_logits.size(2);

    auto p_prime_full = torch::zeros_like(candidate_logits);
    auto probability_ratio_full = torch::zeros_like(candidate_logits);
    auto clamped_sum = torch::full({batch_size, tokens_cand, 1}, 1e-6, torch::dtype(new_logits.dtype()).device(new_logits.device()));

    int max_sram_size;

    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    const int max_threads_div = min(1024, (int)floor((max_sram_size-4096) / (4 * sizeof(float))));
    const int sram_size_div = max_threads_div * 4 * sizeof(float) + 4096;

    dim3 block_size_div(max_threads_div);
    dim3 grid_size_div(batch_size, tokens_cand);

    sampling_kernel_sigmoid<<<grid_size_div, block_size_div, sram_size_div>>>(
            new_logits.data_ptr<float>(),
            candidate_logits.data_ptr<float>(),
            probability_ratio_full.data_ptr<float>(),
            p_prime_full.data_ptr<float>(),
            clamped_sum.data_ptr<float>(),
            batch_size,
            tokens_cand,
            vocab_size,
            max_threads_div
    );
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    auto new_candidate_input_ids = candidate_input_ids.index({Slice(), Slice(-candidate_length, None)});
    auto token_range = torch::arange(candidate_length);
    auto probability_ratio = probability_ratio_full.index({Slice(), torch::arange(candidate_length), new_candidate_input_ids}).squeeze(0).squeeze(1);
    auto r_i = torch::rand_like(probability_ratio);
    auto is_accepted = r_i <= probability_ratio;
    auto n_matches = ((~is_accepted).cumsum(-1) < 1).sum().item<int>();

    if (last_assistant_token_is_eos && n_matches == candidate_length) {
        n_matches -= 1;
        auto valid_tokens = new_candidate_input_ids.index({Slice(), Slice(None, n_matches + 1)});
        return std::make_tuple(valid_tokens, n_matches);
    } else {
        n_matches = std::min(n_matches, max_matches);
        int gamma = std::min((int)candidate_logits.size(1), max_matches);
        // NOTE: new_logits has turned into p at this point (done in the kernel)
        auto p_prime = new_logits.index({Slice(), n_matches, Slice()});
        if (n_matches < gamma) {
            p_prime = p_prime_full.index({Slice(), n_matches, Slice()}) / clamped_sum.index({Slice(), n_matches, Slice()});
        }
        p_prime = torch::clamp_min(p_prime, 0.00001);
        auto t = torch::multinomial(p_prime, 1, false).squeeze(1).index({None, Slice()});
        auto valid_tokens = t;
        if (n_matches > 0) {
            valid_tokens = torch::cat({new_candidate_input_ids.index({Slice(), Slice(None, n_matches)}), t}, -1);
        }
        return std::make_tuple(valid_tokens.to(torch::kLong), n_matches);
    }
}
