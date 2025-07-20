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

#define PROFILING 0
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
#define DQ_BLOCK_TILE_M 32

// A + B = 16384, Codebook: (128 / 8) * 256 * 4 * 2 = 32768
// #define MAX_SHARED_MEMORY_USAGE ((BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K * sizeof(half) + BLOCK_TILE_M * BLOCK_TILE_N * sizeof(float) + CODEBOOK_BUFFERING * (32768 / HOT))
#define MAX_SHARED_MEMORY_USAGE ((BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K * sizeof(half) * 4)
// uint8(1) -> half(2) + codebook
#define DEQUANT_SHARED_MEMORY_USAGE (DQ_BLOCK_TILE_N * DQ_BLOCK_TILE_M * RATIO * 2 + DQ_BLOCK_TILE_N / 4 * ENTRY * RATIO * 2)
#define GEMM_SHARED_MEMORY_USAGE (BLOCK_TILE_M * BLOCK_TILE_K * 2 + BLOCK_TILE_K * RATIO * BLOCK_TILE_N * 2)

// warp tile size: 64*64*16 (4*4 WMMA)
const int WARP_M = 64;
const int WARP_N = 64;
const int WARP_K = 16;
// WMMA tile size: 16*16*16
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
// thread block dim: 32*2*2
const int THREAD_X = 32;
const int THREAD_Y = 2; // warps along N
const int THREAD_Z = 2; // warps along M
constexpr int THREAD_NUM = THREAD_X * THREAD_Y * THREAD_Z;
// fragment number
constexpr int FRAG_A_SIZE = WARP_M / WMMA_M; // 4
constexpr int FRAG_B_SIZE = WARP_N / WMMA_N; // 4
// times of loop
constexpr int K_WARP_COUNT = BLOCK_TILE_K / WARP_K; // 2
constexpr int M_WMMA_COUNT = WARP_M / WMMA_M; // 4
constexpr int N_WMMA_COUNT = WARP_N / WMMA_N; // 4
// 4-stage pipeline multiplier
const int PIPELINE = 8; // 16 bytes = 8 halves


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

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

// For a 4-stage pipeline using cp.async, we load 8 halves at a time instead of 1. 

// Loads a tile of A from global memory into shared memory.
// BLOCK_TILE_M * BLOCK_TILE_K
__device__ void loadSmemA(half *smem, half *A, const int M, const int K, int ko) {
    int thread_id = threadIdx.x + threadIdx.y * THREAD_X + threadIdx.z * THREAD_X * THREAD_Y;
    constexpr int TURNS = (BLOCK_TILE_M * BLOCK_TILE_K) / THREAD_NUM / PIPELINE;
    for (int i = 0; i < TURNS; i++) {
        int row = i * (THREAD_NUM / BLOCK_TILE_K * PIPELINE) + thread_id / (BLOCK_TILE_K / PIPELINE); // i * 32 + thread_id / 4;
        int col = thread_id % (BLOCK_TILE_K / PIPELINE) * PIPELINE; // thread_id % 4 * 8
        
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        int row_out = row / WMMA_M; // row / 16
        int col_out = col / WMMA_K; // col / 16
        int row_in = row % WMMA_M; // row % 16
        int col_in = col % WMMA_K; // col % 16
        int smem_index = row_out * (BLOCK_TILE_K * WMMA_M) + col_out * (WMMA_K * WMMA_M) + row_in * WMMA_K + col_in;
        int A_index = (blockIdx.x * BLOCK_TILE_M + row) * K + ko * BLOCK_TILE_K + col;

        void *ptr = (void *)(smem + smem_index);
        uint32_t smem_ptr;

        // asm: tells the compiler you're inserting inline assembly.
        // The %0, %1, and %2 are placeholders that get filled in by the variables listed later.
        /*
        asm volatile ("assembly code"
              : output operands        <-- between the first and second colon
              : input operands         <-- between the second and third colon
              : clobbered registers);  <-- optional

        .reg .u64 smem_ptr;               // Declare temporary 64-bit register
        cvta.to.shared.u64 smem_ptr, %1;  // Convert ptr to a shared memory address (64-bit)
        cvt.u32.u64 %0, smem_ptr;         // Convert 64-bit to 32-bit address (required for cp.async)

        "r"	General-purpose register
        "l"	Memory operand (pointer or address)
        "n"	Immediate constant (known at compile-time)
        "i"	Arbitrary compile-time constant integer
        "m"	Memory operand (dereferenced pointer)
        */
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));

        // volatile: tells the compiler not to optimize away this code, even if it seems to have no effect. This is critical for operations with side effects like memory loads/stores.
        
        /*
        cp.async.cg.shared.global [dst_shared_ptr], [src_global_ptr], size_in_bytes;
        
        cp.async: "copy asynchronous" — performs data movement using the async pipeline.
        .cg: Cache Global — use the global memory cache path (i.e., via L2).
        .shared.global: Specifies source and destination: from global to shared memory.
        */
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                     :                           // no output
                     : "r"(smem_ptr),            // input operand 0
                       "l"(&A[A_index]),          // input operand 1
                       "n"(16));                 // input operand 2
    }
}

// Loads a tile of B from global memory into shared memory.
// BLOCK_TILE_N * BLOCK_TILE_K
__device__ void loadSmemB(half *smem, half *B, const int N, const int K, int ko) {
    int thread_id = threadIdx.x + threadIdx.y * THREAD_X + threadIdx.z * THREAD_X * THREAD_Y;
    constexpr int TURNS = (BLOCK_TILE_N * BLOCK_TILE_K) / THREAD_NUM / PIPELINE;
    for (int i = 0; i < TURNS; i++) {
        // int row = i * (THREAD_NUM / BLOCK_TILE_K * PIPELINE) + thread_id / (BLOCK_TILE_K / PIPELINE); // i * 32 + thread_id / 4;
        // int col = thread_id % (BLOCK_TILE_K / PIPELINE) * PIPELINE; // thread_id % 4 * 8
        int row = thread_id / 4;
        int col = i * 32 + thread_id % 4 * 8;
        
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        int row_out = row / WMMA_N; // row / 16
        int col_out = col / WMMA_K; // col / 16
        int row_in = row % WMMA_N; // row % 16
        int col_in = col % WMMA_K; // col % 16
        // int smem_index = row_out * (BLOCK_TILE_K * WMMA_N) + col_out * (WMMA_K * WMMA_N) + row_in * WMMA_K + col_in;
        // int B_index = (blockIdx.y * BLOCK_TILE_N + row) * K + ko * BLOCK_TILE_K + col;
        int smem_index = row_out * 8 * 16 * 16 + col_out * 16 * 16 + row_in * 16 + col_in;
        int B_index = (ko * BLOCK_TILE_K + row) * K + blockIdx.y * BLOCK_TILE_N + col;

        void *ptr = (void *)(smem + smem_index);
        uint32_t smem_ptr;

        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));

        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                     :                           // no output
                     : "r"(smem_ptr),            // input operand 0
                       "l"(&B[B_index]),          // input operand 1
                       "n"(16));                 // input operand 2
    }
}

// Stores a tile of C from shared memory into global memory.
// BLOCK_TILE_M * BLOCK_TILE_N
__device__ void storeSmemC(half *C, float *smem, const int M, const int N) {
    int thread_id = threadIdx.x + threadIdx.y * THREAD_X + threadIdx.z * THREAD_X * THREAD_Y;
    constexpr int TURNS = (BLOCK_TILE_M * BLOCK_TILE_N) / THREAD_NUM;
    for (int i = 0; i < TURNS; i++) {
        int row = i * (THREAD_NUM / BLOCK_TILE_N) + thread_id / BLOCK_TILE_N; // i
        int col = thread_id % BLOCK_TILE_N; // thread_id
        
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        int row_out = row / WMMA_M; // row / 16
        int col_out = col / WMMA_N; // col / 16
        int row_in = row % WMMA_M; // row % 16
        int col_in = col % WMMA_N; // col % 16
        int smem_index = row_out * (BLOCK_TILE_N * WMMA_M) + col_out * (WMMA_N * WMMA_M) + row_in * WMMA_N + col_in;
        int C_index = (blockIdx.x * BLOCK_TILE_M + row) * N + blockIdx.y * BLOCK_TILE_N + col;

        C[C_index] = __float2half(smem[smem_index]);
    }
}

// Loads a subtile of As from shared memory to fragment. 
// WARP_M * WARP_K
__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> *frag, half *smem, int ki) {
    for (int i = 0; i < FRAG_A_SIZE; i++) {
        int row = threadIdx.z * WARP_M + i * WMMA_M;
        int col = ki * WMMA_K;
        // layout: [8, 2, 16, 16]
        int row_out = row / WMMA_M; // row / 16
        int col_out = col / WMMA_K; // col / 16
        int row_in = row % WMMA_M; // row % 16
        int col_in = col % WMMA_K; // col % 16
        int smem_index = row_out * (BLOCK_TILE_K * WMMA_M) + col_out * (WMMA_K * WMMA_M) + row_in * WMMA_K + col_in;
        // wmma::load_matrix_sync(fragment, smem_ptr, stride)
        // smem_ptr: pointer to start of tile
        // stride: leading dimension in memory
        nvcuda::wmma::load_matrix_sync(frag[i], smem + smem_index, WMMA_M);
    }
}

// Loads a subtile of Bs from shared memory to fragment. 
// WARP_N * WARP_K
__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> *frag, half *smem, int ki) {
    for (int i = 0; i < FRAG_B_SIZE; i++) {
        // int row = threadIdx.y * WARP_N + i * WMMA_N;
        // int col = ki * WMMA_K;
        int row = ki * WMMA_K;
        int col = threadIdx.y * WARP_N + i * WMMA_N;
        // layout: [2, 8, 16, 16]
        int row_out = row / WMMA_K; // row / 16
        int col_out = col / WMMA_N; // col / 16
        int row_in = row % WMMA_K; // row % 16
        int col_in = col % WMMA_N; // col % 16
        int smem_index = row_out * (BLOCK_TILE_N * WMMA_K) + col_out * (WMMA_K * WMMA_N) + row_in * WMMA_K + col_in;
        // wmma::load_matrix_sync(fragment, smem_ptr, stride)
        // smem_ptr: pointer to start of tile
        // stride: leading dimension in memory
        nvcuda::wmma::load_matrix_sync(frag[i], smem + smem_index, WMMA_N);
    }
}

// Stores a subtile of Cs from fragment to shared memory. 
// WARP_M * WARP_N
__device__ void storeFragC(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> *frag, float *smem) {
    for (int i = 0; i < FRAG_A_SIZE; i++) {
        for (int j = 0; j < FRAG_B_SIZE; j++) {
            int row = threadIdx.z * WARP_M + i * WMMA_M;
            int col = threadIdx.y * WARP_N + j * WMMA_N;
            // layout: [8, 8, 16, 16]
            int row_out = row / WMMA_M; // row / 16
            int col_out = col / WMMA_N; // col / 16
            int row_in = row % WMMA_M; // row % 16
            int col_in = col % WMMA_N; // col % 16
            int smem_index = row_out * (BLOCK_TILE_N * WMMA_M) + col_out * (WMMA_N * WMMA_M) + row_in * WMMA_N + col_in;
            int frag_index = i * FRAG_B_SIZE + j;
            // wmma::load_matrix_sync(smem_ptr, fragment, stride, layout)
            // smem_ptr: pointer to start of tile
            // stride: leading dimension in memory
            // layout: mem_row_major or mem_col_major
            nvcuda::wmma::store_matrix_sync(smem + smem_index, frag[frag_index], WMMA_M, nvcuda::wmma::mem_row_major);
        }
    }
}

// C = alpha * A * B^T + beta * C
// A: M*K, B: N*K, C: M*N
// Test performance using shape M=5376, N=5376, K=2048
__global__ void gemm_kernel(half *A, half *B, half *C, int M, int N, int K) {
    // extern: the size is not defined here, but will be specified when launching the kernel.
    // __shared__: indicates that the variable resides in shared memory, accessible to all threads in the same block.
    // uint8_t: defines a byte-level buffer (uint8_t = 1 byte), which gives you full control over how to split the memory into subregions.
    extern __shared__ uint8_t shared_storage[];

    // half: 16-bit floating point data type (FP-16), <cuda_fp16.h>. Tensor Cores natively support FP16 arithmetic.
    // reinterpret_cast<TYPE *>: Treat the memory at this address as a pointer to TYPE, even if it's not originally that type.
    // 4-stage pipeline
    half *As1 = reinterpret_cast<half *>(shared_storage);
    half *As2 = As1 + BLOCK_TILE_M * BLOCK_TILE_K;
    half *As3 = As2 + BLOCK_TILE_M * BLOCK_TILE_K;
    half *As4 = As3 + BLOCK_TILE_M * BLOCK_TILE_K;
    half *Bs1 = As4 + BLOCK_TILE_M * BLOCK_TILE_K;
    half *Bs2 = Bs1 + BLOCK_TILE_N * BLOCK_TILE_K;
    half *Bs3 = Bs2 + BLOCK_TILE_N * BLOCK_TILE_K;
    half *Bs4 = Bs3 + BLOCK_TILE_N * BLOCK_TILE_K;
    // half * half = float
    float *Cs = reinterpret_cast<float *>(shared_storage);

    // fragment: a small, structured piece of a matrix — like a tile — that is used as input or output to Tensor Core operations.
    // nvcuda::wmma::fragment<role, M, N, K, data_type, layout>
    // Role: matrix_a, matrix_b or accumulator
    // M, N, K: fragment size
    // data_type: usually half for inputs, float for accumulator
    // layout: row_major or col_major
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> Af[FRAG_A_SIZE];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> Bf[FRAG_B_SIZE];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> Cf[FRAG_A_SIZE * FRAG_B_SIZE];
    
    // initialize Cf
    for (int row = 0; row < FRAG_A_SIZE; row++) {
        for (int col = 0; col < FRAG_B_SIZE; col++) {
            nvcuda::wmma::fill_fragment(Cf[row * FRAG_B_SIZE + col], 0.0f);
        }
    }

    // prologue
    loadSmemA(As1, A, M, K, 0);
    loadSmemB(Bs1, B, N, K, 0);
    // tells the GPU that all the cp.async operations you just issued belong to a group, and that group is now done issuing commands.
    asm volatile("cp.async.commit_group;\n" ::);

    loadSmemA(As2, A, M, K, 1);
    loadSmemB(Bs2, B, N, K, 1);
    asm volatile("cp.async.commit_group;\n" ::);

    loadSmemA(As3, A, M, K, 2);
    loadSmemB(Bs3, B, N, K, 2);
    asm volatile("cp.async.commit_group;\n" ::);

    int K_BLOCK_TILE_COUNT = K / BLOCK_TILE_K; // 64
    for (int ko = 0; ko < K_BLOCK_TILE_COUNT - 4; ko += 4) {
        // Wait until 2 previously committed async copy groups have finished.
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 3 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As4, A, M, K, ko + 3);
            loadSmemB(Bs4, B, N, K, ko + 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As1, ki);
            loadFragB(Bf, Bs1, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 4 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As1, A, M, K, ko + 4);
            loadSmemB(Bs1, B, N, K, ko + 4);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As2, ki);
            loadFragB(Bf, Bs2, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 5 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As2, A, M, K, ko + 5);
            loadSmemB(Bs2, B, N, K, ko + 5);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As3, ki);
            loadFragB(Bf, Bs3, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 6 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As3, A, M, K, ko + 6);
            loadSmemB(Bs3, B, N, K, ko + 6);
            //asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As4, ki);
            loadFragB(Bf, Bs4, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }
    }
    {
        int ko = (K_BLOCK_TILE_COUNT / 4 - 1) * 4;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 3 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As4, A, M, K, ko + 3);
            loadSmemB(Bs4, B, N, K, ko + 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As1, ki);
            loadFragB(Bf, Bs1, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        if (ko + 4 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As1, A, M, K, ko + 4);
            loadSmemB(Bs1, B, N, K, ko + 4);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As2, ki);
            loadFragB(Bf, Bs2, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        if (ko + 5 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As2, A, M, K, ko + 5);
            loadSmemB(Bs2, B, N, K, ko + 5);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As3, ki);
            loadFragB(Bf, Bs3, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }

        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        if (ko + 6 < K_BLOCK_TILE_COUNT) {
            loadSmemA(As3, A, M, K, ko + 6);
            loadSmemB(Bs3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < K_WARP_COUNT; ki++) {
            loadFragA(Af, As4, ki);
            loadFragB(Bf, Bs4, ki);
            for (int i = 0; i < M_WMMA_COUNT; i++) {
                for (int j = 0; j < N_WMMA_COUNT; j++) {
                    int Cf_index = i * N_WMMA_COUNT + j;
                    // nvcuda::wmma::mma_sync(result, A, B, accum)
                    // result = A * B + accum
                    nvcuda::wmma::mma_sync(Cf[Cf_index], Af[i], Bf[j], Cf[Cf_index]);
                }
            }
        }
    }
    storeFragC(Cf, Cs);
    __syncthreads();
    storeSmemC(C, Cs, M, N);
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

    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    dim3 block(32, 2, 2);
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
        CHECK_LAST_CUDA_ERROR();
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
    half* codebook_buf = reinterpret_cast<half*>(shmem + DQ_BLOCK_TILE_M * DQ_BLOCK_TILE_N * RATIO * 2);
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
    *(uint32_t*)(&indice[0]) = *(uint32_t*)(&Bq[(Brow + blockIdx.x * DQ_BLOCK_TILE_M) * N + blockIdx.y * DQ_BLOCK_TILE_N + Bcol]);

    // dequant and write to HBM
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 0) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[0]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 1) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[1]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 2) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[2]) * RATIO]);
    *(uint32_t*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol + 3) * RATIO]) = *(uint32_t*)(&codebook_line[((uint32_t)indice[3]) * RATIO]);

    // write back to HBM
    *(int4*)(&B[(Brow + blockIdx.x * DQ_BLOCK_TILE_M) * N * RATIO + (blockIdx.y * DQ_BLOCK_TILE_N + Bcol) * RATIO]) = *(int4*)(&B_buf[(Brow * DQ_BLOCK_TILE_N + Bcol) * RATIO]);
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
    dim3 block(32, 2, 2); // = 128
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    // For dequant kernel
    dim3 dq_grid(M / DQ_BLOCK_TILE_M, N / DQ_BLOCK_TILE_N); // 4096 / 128 blocks. split on N
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
    CHECK_LAST_CUDA_ERROR();
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
    std::cout << "TFLOPS : " << ((2.0 * M * N * K * RATIO) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    return o;
}

