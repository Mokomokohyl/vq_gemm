import importlib
import torch
import numpy as np
import os

import vq_gemm_cuda_cublas_gemm

M = 2048
K = 4096
N = 1280
profiling = (os.getenv('PROFILING', 'FALSE') == 'TRUE')
run_vq_gemm = not (os.getenv('TEST_GEMM', 'FALSE') == 'TRUE')
if not run_vq_gemm:
   M = N = K = 4096
kernel_to_use_str = os.getenv('KERNELS', 'all')
module = importlib.import_module("vq_gemm_cuda_" + kernel_to_use_str)

device = torch.device('cuda')
torch.manual_seed(42)

_MATPLOTLIB = None


def _ensure_matplotlib():
    global _MATPLOTLIB
    if _MATPLOTLIB is None:
        _MATPLOTLIB = importlib.import_module("matplotlib.pyplot")
    return _MATPLOTLIB


def awq_dequantize_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(
        qweight[:, :, None], shifts[None, None, :]
    ).to(torch.int16)
    iweights = iweights.reshape(qweight.shape[0], -1)

    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]
    ).to(torch.int16)
    zeros = zeros.reshape(qzeros.shape[0], -1)

    mask = (1 << bits) - 1
    iweights = torch.bitwise_and(iweights, mask)
    zeros = torch.bitwise_and(zeros, mask)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights.to(scales.dtype) - zeros.to(scales.dtype)) * scales

def vq_gemm_reference(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    w_decoded = awq_dequantize_torch(qweight, scales, qzeros, group_size)
    return vq_gemm_cuda_cublas_gemm.gemm(input, w_decoded)
    # return w_decoded

def gemm_ref(input, w):
    return vq_gemm_cuda_cublas_gemm.gemm(input, w)

def _generate_awq_inputs(m: int, k: int, n: int, device: torch.device):
    group_size = 128
    if k % group_size != 0:
        group_size = k
    if n % 8 != 0:
        raise ValueError("AWQ reference requires N to be divisible by 8.")
    packed_cols = n // 8

    input = torch.randn(m, k, dtype=torch.float16, device=device)
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k, packed_cols),
        dtype=torch.int32,
        device=device,
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // group_size, packed_cols),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand(
        (k // group_size, n), dtype=torch.float16, device=device
    )
    return input, qweight, scales, qzeros, group_size

def main():
    if profiling:
        print("Enter profiling code")
        print(f"  M={M}, K={K}, N={N}")
        print(f"  Device: {device}")
        print("=" * 60)
        input, qweight, scales, qzeros, group_size = _generate_awq_inputs(
            M, K, N, device
        )
        torch.cuda.synchronize()
        output_ref = vq_gemm_reference(input, qweight, scales, qzeros, group_size)
        torch.cuda.synchronize()
        print(f"AWQ reference output mean: {output_ref.mean().item():.6f}")
        return

    print("AWQ GEMM Benchmark")
    print(f"Using kernel VERSION={kernel_to_use_str}")
    print(f"M={M}, K={K}, N={N}")
    print(f"Device: {device}")
    print("=" * 60)

    # 运行 VQ GEMM
    if run_vq_gemm:
        input, qweight, scales, qzeros, group_size = _generate_awq_inputs(
            M, K, N, device
        )
        print(input.shape, input.dtype, input.is_contiguous())
        print(qweight.shape, qweight.dtype, qweight.is_contiguous())
        print(scales.shape, scales.dtype, scales.is_contiguous())
        print(qzeros.shape, qzeros.dtype, qzeros.is_contiguous())
        print(group_size)
        print(module)
        torch.cuda.synchronize()
        output_ref = vq_gemm_reference(input, qweight, scales, qzeros, group_size)
        torch.cuda.synchronize()
        output_cuda = module.e2e_gemm(input, qweight, scales, qzeros, group_size)
        torch.cuda.synchronize()

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

        plt = _ensure_matplotlib()
        plt.imshow(abs_diff_np, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Absolute Error Heatmap")

        # 叠加误差>1的位置为白色点
        mask = abs_diff_np > 1
        ys, xs = np.where(mask)
        plt.scatter(xs, ys, color='white', s=1)  # s=1为点大小，可适当调大

        plt.savefig(f"./M={M}_N={N}_K={K}_err.png")

        outs_cuda = []
        outs_ref = []
        for i in range(5):
            outs_cuda.append(module.e2e_gemm(input, qweight, scales, qzeros, group_size).cpu())
            outs_ref.append(vq_gemm_reference(input, qweight, scales, qzeros, group_size).cpu())

        # 比较 CUDA 输出是否一致
        for i in range(1, 5):
            same = torch.equal(outs_cuda[0], outs_cuda[i])
            print(f"CUDA output run 0 vs {i}: {'一致' if same else '不一致'}")

        # 比较 Reference 输出是否一致
        for i in range(1, 5):
            same = torch.equal(outs_ref[0], outs_ref[i])
            print(f"Reference output run 0 vs {i}: {'一致' if same else '不一致'}")

    else: # test gemm
        print(f"------------ test gemm ------------")
        input = torch.randn(M, K, dtype=torch.float16, device=device)
        w = torch.rand(K, N, dtype=torch.float16, device=device)
        output_cuda = module.gemm(input, w)
        output_ref = gemm_ref(input, w)
        diff = (output_cuda.float() - output_ref.float()).abs().mean().item()
        print(f"Mean absolute difference (CUDA vs Reference): {diff:.6f}")

        abs_diff = (output_cuda.float() - output_ref.float()).abs()

        print(f"Mean absolute difference (CUDA vs Reference): {diff:.6f}")
        max_val, max_idx = abs_diff.max(), abs_diff.argmax()
        max_row, max_col = divmod(max_idx.item(), abs_diff.shape[1])
        print(f"Max abs diff: {max_val.item()}, at ({max_row}, {max_col})")

        abs_diff_np = abs_diff.cpu().numpy()
        plt = _ensure_matplotlib()
        plt.imshow(abs_diff_np, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Absolute Error Heatmap")

        # 叠加误差>1的位置为白色点
        mask = abs_diff_np > 1
        ys, xs = np.where(mask)
        plt.scatter(xs, ys, color='white', s=1)  # s=1为点大小，可适当调大

        plt.savefig(f"./figures/M={M}_N={N}_K={K}_err.png")

        ys, xs = np.where(mask)
        output_cuda_np = output_cuda.cpu().numpy()
        output_ref_np = output_ref.cpu().numpy()

        for y, x in zip(ys, xs):
            print(f"位置 ({y}, {x}): output_cuda={output_cuda_np[y, x]}, output_ref={output_ref_np[y, x]}, abs_diff={abs_diff_np[y, x]}")


if __name__ == "__main__":
    main()