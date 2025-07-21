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

#define PROFILING 1
#define WARP_NUM 4
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_NUM * WARP_SIZE)
#define ENTRY 256
#define RATIO 2
#define RESIDUAL 1
#define HOT 1

#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define BLOCK_TILE_K 32

#define WARP_TILE_M 64
#define WARP_TILE_N 64
#define WARP_TILE_K 16

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 16

#define CODEBOOK_BUFFERING 1

// for dequant kernel
#define DQ_BLOCK_SIZE 1024
#define DQ_BLOCK_TILE_N 128
#define DQ_BLOCK_TILE_K 32

// A + B = 16384, Codebook: (128 / 8) * 256 * 4 * 2 = 32768
#define MAX_SHARED_MEMORY_USAGE (16384 + CODEBOOK_BUFFERING * (32768 / HOT))
// uint8(1) -> half(2) + codebook
#define DEQUANT_SHARED_MEMORY_USAGE (DQ_BLOCK_TILE_N * DQ_BLOCK_TILE_K * RATIO * 2 + DQ_BLOCK_TILE_N / 4 * ENTRY * RATIO * 2)
#define GEMM_SHARED_MEMORY_USAGE (BLOCK_TILE_M * BLOCK_TILE_K * 2 + BLOCK_TILE_K * RATIO * BLOCK_TILE_N * 2)
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
    for (int i = 0; i < ((BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_SIZE) / 8; i++) {
        int row = i * 32 + threadIdx.x / 4;
        int col = 8 * (threadIdx.x % 4);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            ::
            "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K)), "l"(&A[(blockIdx.x * BLOCK_TILE_M + row) * k + ko * BLOCK_TILE_K + col])
        );
    }
}

__device__ void loadShmemB(half* shmem, half *B, int k, int n, int ko) {
    for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / (WARP_SIZE * WARP_NUM) / 2; i++) {
        int row = i * 2 + threadIdx.x / 64;
        int col = 2 * (threadIdx.x % 64);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            ::
            "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N))), "l"(&B[(ko * BLOCK_TILE_K + row) * n + blockIdx.y * BLOCK_TILE_N + col])
        );                
    }
}

__device__ void loadFragA_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 4; i++) {       // Warp do 64x16, 16x16 a time, so 4 times
        // for (int j = 0; j < 4; j++) {   // for every 16x16, every thread load 4 1x2 data
        //     int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + (j / 2) * 8 + (lane_id / 4);
        //     int col = ki * WMMA_TILE_K + (j % 2) * 8 + (lane_id % 4) * 2;
        //     frag[i * 4 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K);
        // }
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
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // for (int i = 0; i < 8; i++) {       // Warp do 16x64, 16x8 a time, so 8 times
    //     for (int j = 0; j < 2; j++) {   // for every 16x8, every thread load 2 1x2 data
    //         int row = ki * WARP_TILE_K + j * 8 + (lane_id / 4);
    //         int col = warp_id_y * WARP_TILE_N + i * 8 + (lane_id % 4) * 2;
    //         frag[i * 2 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N));
    //     }
    //     // Can directly use ldmatrix.trans
    //     asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2]) : "r"(frag[i * 2]));
    //     asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2 + 1]) : "r"(frag[i * 2]));
    // }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
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
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 4; i++) {           // 4 rows
        for (int j = 0; j < 8; j++) {       // 8 cols
            for (int k = 0; k < 2; k++) {   // 2 frags
                int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + k * 8 + (lane_id / 4);
                int col = warp_id_y * WARP_TILE_N + j * 8 + (lane_id % 4) * 2;
                *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N) + (row % WMMA_TILE_M) * (WMMA_TILE_N) + (col % WMMA_TILE_N)) = 
                frag[i * 8 * 2 + j * 2 + k];
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

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

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
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
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
    // cudaFuncSetAttribute(e2e_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
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

    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
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
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
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

// each threadBlock load 32 x 128 uint8, each thread dequant 4 uint8 
__global__ void dequant_kernel(
    uint8_t* Bq, // [K, N]
    half* _codebook, // [N / 4, ENTRY * RATIO]
    half* B, // [K, N * RATIO]
    int N, int K
)
{    
    extern __shared__ uint8_t shmem[];
    uint8_t indice[4];
    half* B_buf = reinterpret_cast<half*>(shmem);
    // codebook size: [DQ_BLOCK_TILE_N / 4 (=32) ,ENTRY * RATIO] (RATIO = 2)
    half* codebook_buf = reinterpret_cast<half*>(shmem + DQ_BLOCK_TILE_K * DQ_BLOCK_TILE_N * RATIO * 2);
    uint32_t Brow = threadIdx.x % 32; // [0, 31]
    uint32_t Bcol = (threadIdx.x / 32) * 4; // 4 * [0, 31]
    // Load Codebook to shmem

    uint32_t codebook_begin_row = blockIdx.y * DQ_BLOCK_TILE_N / 4;
    // uint32_t iters_to_load = ((32 * ENTRY * RATIO) / 8) / BLOCK_SIZE; // = 2
    uint32_t load_cols = (ENTRY * RATIO) / 8; // =64
    uint32_t load_rows = DQ_BLOCK_SIZE / load_cols; // =16

    asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
    :
    : "r"(shmem_uint32_t(&codebook_buf[(threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8])),
        "l"(&_codebook[(codebook_begin_row + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8])
    );
    // *(int4*)(&codebook_buf[(threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8]) = 
    // *(int4*)(&_codebook[(codebook_begin_row + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8]);

    asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
    :
    : "r"(shmem_uint32_t(&codebook_buf[(load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8])),
        "l"(&_codebook[(codebook_begin_row + load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8])
    );
    // *(int4*)(&codebook_buf[(load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8]) = 
    // *(int4*)(&_codebook[(codebook_begin_row + load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8]);

    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    half* codebook_line = codebook_buf + (Bcol / 4) * ENTRY * RATIO;

    // Load Bq indice to reg.
    *(uint32_t*)(&indice[0]) = *(uint32_t*)(&Bq[(Brow + blockIdx.x * DQ_BLOCK_TILE_K) * N + blockIdx.y * DQ_BLOCK_TILE_N + Bcol]);

    // dequant and write to HBM
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 0) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[0]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 1) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[1]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 2) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[2]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 3) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[3]) * RATIO]);

    // write back to HBM
    *(int4*)(&B[(Brow + blockIdx.x * DQ_BLOCK_TILE_K) * N * RATIO + (blockIdx.y * DQ_BLOCK_TILE_N + Bcol) * RATIO]) = *(int4*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol) * RATIO]);
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
    // cudaFuncSetAttribute(e2e_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
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
    torch::Tensor B = torch::full({K, N * RATIO}, 0, options);
    half* B_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());

    dim3 grid(M / BLOCK_TILE_M, N / (BLOCK_TILE_N / RATIO));
    dim3 block(BLOCK_SIZE); // = 128
    // For dequant kernel
    dim3 dq_grid(K / DQ_BLOCK_TILE_K, N / DQ_BLOCK_TILE_N); // 4096 / 128 blocks. split on N
    dim3 dq_block(DQ_BLOCK_SIZE); // = 1024
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        dequant_kernel<<<dq_grid, dq_block, DEQUANT_SHARED_MEMORY_USAGE>>>(
            w_ptr,
            codebook_ptr,
            B_ptr,
            N, K
        );
        gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            B_ptr,
            o_ptr,
            M, N * RATIO, K
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        dequant_kernel<<<dq_grid, dq_block, DEQUANT_SHARED_MEMORY_USAGE>>>(
            w_ptr,
            codebook_ptr,
            B_ptr,
            N, K
        );
        gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            B_ptr,
            o_ptr,
            M, N * RATIO, K
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

