import torch
import numpy as np
import os

import vq_gemm_cuda

M = 128
K = 128
N = 32
ENTRY = 256
RATIO = 4

device = torch.device('cuda')
torch.manual_seed(42)

import torch

def vq_gemm_reference(input, w, codebook):
    # input: [M, K]
    # w: [K, N]，每个元素为uint8
    # codebook: [N // 2, ENTRY, RATIO]
    K, N = w.shape
    M = input.shape[0]

    # 构造解码后的权重
    w_decoded = torch.empty(K, N * RATIO, dtype=torch.float16, device=input.device)
    for k in range(K):
        for no in range(N // 32):
            for ni in range(32):
                row_idx = ni // 2 + no * 32
                entry_idx = w[k, no * 32 + ni].item()
                # 查码本，得到[RATIO]个half
                w_decoded[k, (no * 32 + ni) * RATIO : (no * 32 + ni + 1) * RATIO] = codebook[row_idx, entry_idx]
    # GEMM
    output = torch.matmul(input, w_decoded)
    return output

def main():
    print(f"VQ GEMM Benchmark")
    print(f"  M={M}, K={K}, N={N}, ENTRY={ENTRY}, RATIO={RATIO}")
    print(f"  Device: {device}")
    print("=" * 60)

    input = torch.randn(M, K, dtype=torch.float16, device=device)
    w = torch.randint(0, ENTRY, (K, N), dtype=torch.uint8, device=device)
    codebook = torch.randn(N // 2, ENTRY, RATIO, dtype=torch.float16, device=device)

    # 运行 VQ GEMM
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    output_cuda = vq_gemm_cuda.e2e_gemm(input, w, codebook)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    output_ref = vq_gemm_reference(input, w, codebook)

    print(f"VQ GEMM finished in {elapsed_time:.3f} ms")
    print(f"VQ GEMM output shape:{output_cuda.shape}")
    print("Row mean of VQ GEMM output (Reference):", output_ref.mean(dim=1))
    print("Row mean of VQ GEMM output (CUDA):", output_cuda.mean(dim=1))
    diff = (output_cuda.float() - output_ref.float()).abs().mean().item()
    print(f"Mean absolute difference (CUDA vs Reference): {diff:.6f}")

if __name__ == "__main__":
    main()