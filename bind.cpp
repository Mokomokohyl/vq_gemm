#include <torch/extension.h>

torch::Tensor e2e_gemm(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    int group_size
);
torch::Tensor gemm(torch::Tensor input, torch::Tensor w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("e2e_gemm", &e2e_gemm, "VQ GEMM");
    m.def("gemm", &gemm, "Normal GEMM");
}