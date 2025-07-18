import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import vq_gemm_cuda

M = 4096
K = 4096
N = 1024
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
    RATIO = codebook.shape[2]

    # 计算每个列对应的分块行号
    row_idx = torch.arange(N, device=w.device) // 2  # [N]
    # 展开为 [K, N]，每个元素是分块行号
    row_idx_expand = row_idx.unsqueeze(0).expand(K, N)
    # entry_idx就是w本身
    entry_idx = w.long()  # [K, N]

    # 用高级索引一次性查码本，得到 [K, N, RATIO]
    w_decoded = codebook[row_idx_expand, entry_idx]  # [K, N, RATIO]
    # reshape为 [K, N * RATIO]
    w_decoded = w_decoded.reshape(K, N * RATIO)
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
    abs_diff = (output_cuda.float() - output_ref.float()).abs()
    diff = (output_cuda.float() - output_ref.float()).abs().mean().item()
    print(f"Mean absolute difference (CUDA vs Reference): {diff:.6f}")
    max_val, max_idx = abs_diff.max(), abs_diff.argmax()
    max_row, max_col = divmod(max_idx.item(), abs_diff.shape[1])
    print(f"Max abs diff: {max_val.item()}, at ({max_row}, {max_col})")

    abs_diff_np = abs_diff.cpu().numpy()
    plt.imshow(abs_diff_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Absolute Error Heatmap")

    # 叠加误差>1的位置为白色点
    mask = abs_diff_np > 1
    ys, xs = np.where(mask)
    plt.scatter(xs, ys, color='white', s=1)  # s=1为点大小，可适当调大

    plt.savefig(f"./figures/M={M}_N={N}_K={K}_err.png")

    outs_cuda = []
    outs_ref = []
    for i in range(5):
        outs_cuda.append(vq_gemm_cuda.e2e_gemm(input, w, codebook).cpu())
        outs_ref.append(vq_gemm_reference(input, w, codebook).cpu())


    # 比较 CUDA 输出是否一致
    for i in range(1, 5):
        same = torch.equal(outs_cuda[0], outs_cuda[i])
        print(f"CUDA output run 0 vs {i}: {'一致' if same else '不一致'}")

    # 比较 Reference 输出是否一致
    for i in range(1, 5):
        same = torch.equal(outs_ref[0], outs_ref[i])
        print(f"Reference output run 0 vs {i}: {'一致' if same else '不一致'}")

if __name__ == "__main__":
    main()