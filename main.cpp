#include <torch/extension.h>
std::tuple<torch::Tensor, int> sampling_cuda(torch::Tensor candidate_input_ids, torch::Tensor candidate_logits, int candidate_length, torch::Tensor new_logits, bool last_assistant_token_is_eos, int max_matches);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sampling_cuda", torch::wrap_pybind_function(sampling_cuda), "sampling_cuda");
}


