#include <torch/extension.h>

torch::Tensor e2e_gemm(torch::Tensor input, torch::Tensor w, torch::Tensor codebook);
torch::Tensor e2e_gemm_rq(torch::Tensor input, torch::Tensor w, torch::Tensor codebook, torch::Tensor w_r, torch::Tensor codebook_r);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("e2e_gemm", &e2e_gemm, "VQ GEMM");
}