#include <torch/extension.h>

torch::Tensor e2e_gemm(torch::Tensor input, torch::Tensor w, torch::Tensor codebook);
torch::Tensor gemm(torch::Tensor input, torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("e2e_gemm", &e2e_gemm, "VQ GEMM");
    m.def("gemm", &gemm, "Normal GEMM");
}