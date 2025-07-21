#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <cublas_v2.h>
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "mma.h"
#include <random>


// cublas gemm
torch::Tensor gemm(
    torch::Tensor input,
    torch::Tensor w
)
{
    auto M = input.size(0);
    auto K = input.size(1);
    auto N = w.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({M, N}, 0, options);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

    half* w_ptr = reinterpret_cast<half*>(w.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w_ptr, CUDA_R_16F, N,
        input_ptr, CUDA_R_16F, K,
        &beta,
        o_ptr, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT);
    return o;
}

// Should not be called
torch::Tensor e2e_gemm(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
)
{
    auto M = input.size(0);
    auto N = w.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({M, N}, 0, options);
    return o;
}

