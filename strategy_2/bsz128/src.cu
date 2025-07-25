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

// A + B = 16384, Codebook: (128 / 8) * 256 * 4 * 2 = 32768
#define MAX_SHARED_MEMORY_USAGE (16384 * 2 + CODEBOOK_BUFFERING * (32768 / HOT))
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

__device__ void storeC(half* C, uint32_t* frag, int m, int n) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 0) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * 8 + j) * 2 + 0]);
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 8) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * 8 + j) * 2 + 1]);
        }
    }
}

__device__ void dequantToShmemB(half* shmem, uint8_t* B_q, half* codebook, half* codebook_shmem, int k, int n, int ko) {
    // 32x64 uint8, every thread load 16 uint8 indices
    uint32_t local_id = (threadIdx.x % 4) * 4;

    uint8_t indices[16];
    *(uint64_t*)(&indices[0]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K) * n + blockIdx.y * (BLOCK_TILE_N / RATIO) + (threadIdx.x / 4) * n + (threadIdx.x % 4) * 16]);
    *(uint64_t*)(&indices[8]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K) * n + blockIdx.y * (BLOCK_TILE_N / RATIO) + (threadIdx.x / 4) * n + (threadIdx.x % 4) * 16 + 8]);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        *(uint32_t*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4 * 16 + i) * 2 / 16 * (16 * 16) + (threadIdx.x / 4) % 16 * 16 + (threadIdx.x % 4 * 8 + i) * 2 % 16]) = *(uint32_t*)(&codebook_shmem[(local_id + i / 4) * 256 * RATIO + ((uint32_t) indices[i]) * RATIO]);
    }
}

__device__ void load_codebook(
    half* shmem,
    half* codebook
)
{
    uint32_t codebook_begin_row = blockIdx.y * 16;
    // Assuming HOT is less than 16
    uint32_t iters_to_load = ((16 * ENTRY * RATIO / HOT) / 8) / BLOCK_SIZE;
    uint32_t load_cols = (ENTRY * RATIO / HOT) / 8;
    uint32_t load_rows = BLOCK_SIZE / load_cols;

    #pragma unroll
    for (int i = 0; i < iters_to_load; i++) {
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shmem_uint32_t(&shmem[(i * load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO / HOT) + (threadIdx.x % load_cols) * 8])),
          "l"(&codebook[(codebook_begin_row + i * load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO) + (threadIdx.x % load_cols) * 8])
        );
    }
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
    half *codebook_buf = reinterpret_cast<half*>(shmem + (BLOCK_TILE_M * BLOCK_TILE_K + BLOCK_TILE_K * BLOCK_TILE_N) * sizeof(half));

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

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
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
        __syncthreads();
    }
    storeC(_o, C_frags, M, N * RATIO);    
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
    half *A2 = reinterpret_cast<half*>(shmem + (BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K * sizeof(half));
    half *B2 = reinterpret_cast<half*>(shmem + (2 * BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};
    
    // prologue: ko = 0. fill buffer 1
    loadShmemA(A1, _input, M, K, 0);
    loadShmemB(B1, _w, K, N, 0);
    asm volatile("cp.async.commit_group; \n" ::);
    __syncthreads();

    // main pipeline: 1, 2, ..., K / BLOCK_TILE_K / 2 * 2 - 2
    for (int ko = 1; ko < (K / BLOCK_TILE_K) / 2 * 2 - 2; ko += 2) {

        // launch buffer 2 loading
        loadShmemA(A2, _input, M, K, ko); // cp.async
        loadShmemB(B2, _w, K, N, ko);
        asm volatile("cp.async.commit_group; \n" ::);

        // wait for buffer 1
        asm volatile("cp.async.wait_group %0; \n" ::"n"(1));
        __syncthreads();

        // consume buffer 1
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A1, ki);
            loadFragB_mma(B_frags, B1, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
        // launch buffer 1 loading
        loadShmemA(A1, _input, M, K, ko + 1); // cp.async
        loadShmemB(B1, _w, K, N, ko + 1);
        asm volatile("cp.async.commit_group; \n" ::);

        // wait for buffer 2
        asm volatile("cp.async.wait_group %0; \n" ::"n"(1));
        __syncthreads();

        // consume buffer 2
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A2, ki);
            loadFragB_mma(B_frags, B2, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
    }

    // epilogue: ko = K / BLOCK_TILE_K / 2 * 2 - 1, ..., K / BLOCK_TILE_K - 1
    int ko = (K / BLOCK_TILE_K) / 2 * 2 - 1;
    // launch buffer 2 loading
    loadShmemA(A2, _input, M, K, ko); // cp.async
    loadShmemB(B2, _w, K, N, ko);
    asm volatile("cp.async.commit_group; \n" ::);
    // wait for buffer 1
    asm volatile("cp.async.wait_group %0; \n" ::"n"(1));
    __syncthreads();

    // consume buffer 1
    for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
        loadFragA_mma(A_frags, A1, ki);
        loadFragB_mma(B_frags, B1, ki);
        for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
            for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
            }
        }
    }
    if ((ko + 1) < K / BLOCK_TILE_K) {
        // launch buffer 1 loading
        loadShmemA(A1, _input, M, K, ko + 1); // cp.async
        loadShmemB(B1, _w, K, N, ko + 1);
        asm volatile("cp.async.commit_group; \n" ::);
        // wait for buffer 2
        asm volatile("cp.async.wait_group %0; \n" ::"n"(1));
        __syncthreads();
        // consume buffer 2
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A2, ki);
            loadFragB_mma(B_frags, B2, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
        // wait for buffer 1
        asm volatile("cp.async.wait_group %0; \n" ::"n"(0));
        __syncthreads();
        // consume buffer 1
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A1, ki);
            loadFragB_mma(B_frags, B1, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
    } else {
        // wait for buffer 2
        asm volatile("cp.async.wait_group %0; \n" ::"n"(0));
        __syncthreads();
        // consume buffer 2
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A2, ki);
            loadFragB_mma(B_frags, B2, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
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
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
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
    std::cout << "TFLOPS : " << ((2.0 * M * N * K * RATIO) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    return o;
}
