#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "mma.h"
#include <random>

#define PROFILING 1
#define WARP_NUM 16 // 16 * 32 = 512
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_NUM * WARP_SIZE)
#define ENTRY 256
#define RATIO 2
#define RESIDUAL 1
#define HOT 1

#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define BLOCK_TILE_K 32

#define WARP_TILE_M 32 // 128 / 32 = 4. 4 * 4 = 16 warps
#define WARP_TILE_N 32
#define WARP_TILE_K 16

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 16

#define CODEBOOK_BUFFERING 1

// A + B = 16384, Codebook: (128 / 8) * 256 * 4 * 2 = 32768
#define MAX_SHARED_MEMORY_USAGE (16384 + CODEBOOK_BUFFERING * (32768 / HOT))
__device__ __forceinline__ uint32_t shmem_uint32_t(const void* shmem_ptr) {
    uint32_t addr;
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(shmem_ptr)
    );
    return addr;
}

__device__ void loadShmemA(half* shmem, half *A, int m, int k, int ko) {
    // 512 threads load 128 x 32 halves. each thread load 8 halves(16 bytes), a single cp.async ..., 16
    int row = threadIdx.x / 4; // [0, 127]
    int col = 8 * (threadIdx.x % 4);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        ::
        "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K)), "l"(&A[(blockIdx.x * BLOCK_TILE_M + row) * k + ko * BLOCK_TILE_K + col])
    );
}

__device__ void loadShmemB(half* shmem, half *B, int k, int n, int ko) {
    // 512 threads load 32 x 128 halves. each thread load 8 halves(16 bytes), a single cp.async ..., 16
    int row = threadIdx.x / 16;
    int col = 8 * (threadIdx.x % 16);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        ::
        "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N))), "l"(&B[(ko * BLOCK_TILE_K + row) * n + blockIdx.y * BLOCK_TILE_N + col])
    );                
}

__device__ void loadFragA_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / (BLOCK_TILE_M / WARP_TILE_M); // 0, 1, 2, 3
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < (WARP_TILE_M / WMMA_TILE_M); i++) {
        int row = warp_id_x * WARP_TILE_M + i * 16 + (lane_id % 16);
        int col = ki * WARP_TILE_K + (lane_id / 16) * 8;
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
            : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K))
        );
    }
}

__device__ void loadFragB_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % (BLOCK_TILE_N / WARP_TILE_N); // 0, 1, 2, 3
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < (WARP_TILE_N / WMMA_TILE_N); i++) {
        int row = ki * WARP_TILE_K + (lane_id % 16);
        int col = warp_id_y * WARP_TILE_N + i * 16 + (lane_id / 16) * 8;
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
            : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N)))
        );
    }
}

__device__ void compute_mma(uint32_t* A, uint32_t* B, uint32_t* C) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1])
    );
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[2]), "r"(B[3]),
          "r"(C[2]), "r"(C[3])
    );
}

__device__ void storeFragC_mma(half* shmem, uint32_t* frag) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / (BLOCK_TILE_M / WARP_TILE_M);
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % (BLOCK_TILE_N / WARP_TILE_N);
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < (WARP_TILE_M / 16); i++) {           // 4 rows -> WARP_TILE_M / 16 rows (=2)
        for (int j = 0; j < (WARP_TILE_N / 8); j++) {       // 8 cols -> WARP_TILE_N / 8 rows (=4)
            for (int k = 0; k < 2; k++) {   // 2 frags
                int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + k * 8 + (lane_id / 4);
                int col = warp_id_y * WARP_TILE_N + j * 8 + (lane_id % 4) * 2;
                *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N) + (row % WMMA_TILE_M) * (WMMA_TILE_N) + (col % WMMA_TILE_N)) = 
                frag[i * (WARP_TILE_N / 8) * 2 + j * 2 + k];
            }
        }
    }
}

__device__ void storeShmemC(half *C, half* shmem, int m, int n) {
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_N) / (WARP_SIZE * WARP_NUM); i++) {
        int row = i * ((WARP_SIZE * WARP_NUM) / BLOCK_TILE_M) + threadIdx.x / BLOCK_TILE_N;
        int col = threadIdx.x % BLOCK_TILE_N;
        C[(blockIdx.x * BLOCK_TILE_M + row) * n + (blockIdx.y * BLOCK_TILE_N + col)] = 
        shmem[(row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N) + (row % WMMA_TILE_M) * (WMMA_TILE_N) + col % WMMA_TILE_N];
    }
}

__device__ void storeC(half* C, uint32_t* frag, int m, int n) {
    // may be problematic
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / (WARP_TILE_M / WMMA_TILE_M);
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % (WARP_TILE_N / WMMA_TILE_N);
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < (WARP_TILE_M / 16); i++) {
        #pragma unroll
        for (int j = 0; j < (WARP_TILE_N / 8); j++) {
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 0) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * (WARP_TILE_N / 8) + j) * 2 + 0]);
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 8) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * (WARP_TILE_N / 8) + j) * 2 + 1]);
        }
    }
}

__device__ void dequantToShmemB(half* shmem, uint8_t* B_q, half* codebook, half* codebook_shmem, int k, int n, int ko) {
    // 32x64 uint8, 512 threads, every thread dequant 4 uint8_t
    int tid = threadIdx.x;
    uint32_t row = tid / 16; // [0, 32] 
    uint32_t col = tid % 16 * 4; // 4 * [0, 15]
    uint32_t local_idx = col / 4;

    uint8_t indices[4];
    *(uint32_t*)(&indices[0]) = *(uint32_t*)(&B_q[(ko * BLOCK_TILE_K + row) * n + blockIdx.y * (BLOCK_TILE_N / RATIO) + col]);
    *(uint32_t*)(&shmem[row / 16 * (8 * 16 * 16) + col * RATIO / 16 * (16 * 16) + row % 16 * 16 + (col * RATIO) % 16]) = *(uint32_t*)(&codebook_shmem[local_idx * 256 * RATIO + ((uint32_t) indices[0]) * RATIO]);
    *(uint32_t*)(&shmem[row / 16 * (8 * 16 * 16) + (col + 1) * RATIO / 16 * (16 * 16) + row % 16 * 16 + ((col + 1) * RATIO) % 16]) = *(uint32_t*)(&codebook_shmem[local_idx * 256 * RATIO + ((uint32_t) indices[1]) * RATIO]);
    *(uint32_t*)(&shmem[row / 16 * (8 * 16 * 16) + (col + 2) * RATIO / 16 * (16 * 16) + row % 16 * 16 + ((col + 2) * RATIO) % 16]) = *(uint32_t*)(&codebook_shmem[local_idx * 256 * RATIO + ((uint32_t) indices[2]) * RATIO]);
    *(uint32_t*)(&shmem[row / 16 * (8 * 16 * 16) + (col + 3) * RATIO / 16 * (16 * 16) + row % 16 * 16 + ((col + 3) * RATIO) % 16]) = *(uint32_t*)(&codebook_shmem[local_idx * 256 * RATIO + ((uint32_t) indices[3]) * RATIO]);
}

__device__ void load_codebook(
    half* shmem,
    half* codebook
)
{
    int tid = threadIdx.x;

    uint32_t codebook_begin_row = blockIdx.y * 16;
    // Assuming HOT is less than 16
    // uint32_t iters_to_load = ((16 * ENTRY * RATIO / HOT) / 8) / BLOCK_SIZE; // = 2
    uint32_t load_cols = (ENTRY * RATIO) / 8; // = 64
    uint32_t load_rows = (WARP_NUM * WARP_SIZE) / load_cols; // = 8

    asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n" 
    :
    :   "r"(shmem_uint32_t(&shmem[(tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])), 
        "l"(&codebook[(codebook_begin_row + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])
    );
    asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n" 
    :
    :   "r"(shmem_uint32_t(&shmem[(load_rows + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])),
        "l"(&codebook[(load_rows + codebook_begin_row + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])
    );
}

__global__ void e2e_gemm_kernel(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A1 = reinterpret_cast<half*>(shmem);
    half *B1 = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);
    half *codebook_buf = reinterpret_cast<half*>(shmem + (BLOCK_TILE_M * BLOCK_TILE_K + BLOCK_TILE_K * BLOCK_TILE_N) * sizeof(half));

    uint32_t A_frags[8];
    uint32_t B_frags[8];
    uint32_t C_frags[16] = {0};

    // Load codebook
    load_codebook(codebook_buf, _codebook);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        loadShmemA(A1, _input, M, K, ko);
        dequantToShmemB(B1, _w, _codebook, codebook_buf, K, N, ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A1, ki);
            loadFragB_mma(B_frags, B1, ki);
            // dequantToRegB(B_frags, _w, _codebook, codebook_buf, K, N, ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * (WARP_TILE_M / WMMA_TILE_M) + nn) * 4]);
                }
            }
        }
        __syncthreads();
    }
    storeFragC_mma(C_buf, C_frags);
    __syncthreads();
    storeShmemC(_o, C_buf, M, N * RATIO);   
}

torch::Tensor e2e_gemm(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
)
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
#endif
    cudaFuncSetAttribute(e2e_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    // Assuming M is padded to 128, pad at torch level.

    auto M = input.size(0);
    auto K = input.size(1);
    auto N = w.size(1);
    std::cout << M << " " << K << " " << N << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({M, N * RATIO}, 0, options);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

    uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    dim3 grid(M / BLOCK_TILE_M, N / (BLOCK_TILE_N / RATIO));
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        e2e_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            o_ptr,
            M, N, K
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        e2e_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            o_ptr,
            M, N, K
        );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
    std::cout << "TFLOPS : " << ((2.0 * M * N * K * RATIO) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    return o;
}

__global__ void gemm_kernel(
    half* _input,
    half* _w,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A1 = reinterpret_cast<half*>(shmem);
    half *B1 = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);

    uint32_t A_frags[8];
    uint32_t B_frags[8];
    uint32_t C_frags[16] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        loadShmemA(A1, _input, M, K, ko); // cp.async
        loadShmemB(B1, _w, K, N, ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A1, ki);
            loadFragB_mma(B_frags, B1, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * (WARP_TILE_M / WMMA_TILE_M) + nn) * 4]);
                }
            }
        }
        __syncthreads();
    }
    storeFragC_mma(C_buf, C_frags);
    __syncthreads();
    storeShmemC(_o, C_buf, M, N);   
}

torch::Tensor gemm(
    torch::Tensor input,
    torch::Tensor w
)
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
#endif
    cudaFuncSetAttribute(e2e_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    // Assuming M is padded to 128, pad at torch level.

    auto M = input.size(0);
    auto K = input.size(1);
    auto N = w.size(1);
    std::cout << M << " " << K << " " << N << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({M, N}, 0, options);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

    half* w_ptr = reinterpret_cast<half*>(w.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            o_ptr,
            M, N, K
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            o_ptr,
            M, N, K
        );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
    std::cout << "TFLOPS : " << ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    return o;
}
