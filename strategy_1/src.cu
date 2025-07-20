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

const int wmmaM = 16, wmmaN = 16, wmmaK = 16;

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

__device__ void loadSmemA(half *SA, half *dA, int M, int K, int ko)
{
    //仍然每次load 32个half, sizeof(half) = 2, 用cp.async一次load 16个字节那么只需要load4次
    // load 128 * 32, 8行2列块
    int tid = 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x;
    for (int i = 0; i < 4; ++i)
    {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;

        void *ptr = (void *)(SA + row / 16 * (2 * 16 * 16) + col / 16 * 16 * 16 + row % 16 * 16 + col % 16);
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr)
        );

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
            "l"(&dA[(blockIdx.y * 128 + row) * K + (ko * 32 + col)]),
            "n"(16)
        );
    }
}

__device__ void loadSmemB(half *SB, half *dB, int K, int N, int ko)
{
    int tid = 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x;
    // load 32 * 128, 2行8列块
    for (int i = 0; i < 4; ++i)
    {
        int row = tid / 4;
        int col = tid % 4 * 8 + i * 32;

        void *ptr = (void *)(SB + row / 16 * (8 * 16 * 16) + col / 16 * 16 * 16 + row % 16 * 16 + col % 16);
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr)
        );

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
            "l"(&dB[(row + ko * 32) * N + blockIdx.x * 128 + col]),
            "n"(16)
        );
    }
}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major>* fragA, half *SmemA, int ki)
{
    int mi = 4 * threadIdx.y;
    for (int i = 0; i < 4; i++)
    {
        half *mptr = SmemA + (mi + i) * 2 * 16 * 16 + ki * 16 * 16;
        nvcuda::wmma::load_matrix_sync(fragA[i], mptr, 16);
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major>* fragB, half *SmemB, int ki)
{
    int ni = 4 * threadIdx.z;
    for (int i = 0; i < 4; i++)
    {
        half *mptr = SmemB + ki * 8 * 16 * 16 + (ni + i) * 16 * 16;
        nvcuda::wmma::load_matrix_sync(fragB[i], mptr, 16);
    }
}

__device__ void storeAccum(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float>* Accum, float *SmemC)
{
    //store 64 x 64
    for (int i = 0; i < 4; ++i) 
    {
        for (int j = 0; j < 4; ++j) 
        {
            int row = threadIdx.y * 64 + i * 16;
            int col = threadIdx.z * 64 + j * 16;
            nvcuda::wmma::store_matrix_sync(SmemC + row / 16 * 8 * 16 * 16 + col / 16 * 16 * 16, Accum[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

__device__ void storeSmemC(float *SmemC, half*dC, int M, int N)
{
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        dC[(row + blockIdx.y * 128) * N + blockIdx.x * 128 + col] = __float2half(SmemC[row / 16 * 8 * 16 * 16 + col / 16 * 16 * 16 + row % 16 * 16 + col % 16]);
    }
}

template <int BM, int BN, int TM, int TN, int TK>
__global__ void gemm_kernel(half *dA, half *dB, half *dC, int M, int N, int K)
{
    extern __shared__ uint8_t smem[];
    float *SmemC = (float *)(smem);
    half *SA1 = (half *)smem;
    half *SA2 = SA1 + BM * TK;
    half *SA3 = SA2 + BM * TK;
    half *SA4 = SA3 + BM * TK;
    half *SB1 = SA4 + BN * TK;
    half *SB2 = SB1 + BN * TK;
    half *SB3 = SB2 + BN * TK;
    half *SB4 = SB3 + BN * TK;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragA[TM / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> fragB[TN / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[TM / wmmaM * TN / wmmaN];

    for (int mi = 0; mi < BM / wmmaM; mi++)
    {
        for (int ni = 0; ni < BN / wmmaN; ni++) 
        {
            nvcuda::wmma::fill_fragment(Accum[mi * BN / wmmaN + ni], 0.0);
        }
    }
    // Each thread block: BM x BN x TK

    // prologue
    // [0, 1, 2]
    loadSmemA(SA1, dA, M, K, 0);
    loadSmemB(SB1, dB, K, N, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    loadSmemA(SA2, dA, M, K, 1);
    loadSmemB(SB2, dB, K, N, 1);
    asm volatile("cp.async.commit_group;\n" ::);

    loadSmemA(SA3, dA, M, K, 2);
    loadSmemB(SB3, dB, K, N, 2);
    asm volatile("cp.async.commit_group;\n" ::);

    // main pipeline
    // ko_max = (K / TK - 8) / 4 * 4;
    // [3 , ... , (K / TK / 4) * 4 - 8 + 6] = (K / TK / 4 * 4 - 2)]
    int ko_max = (K / TK - 8) / 4 * 4;
    for (int ko = 0; ko <= ko_max; ko += 4)
    {
        // wait for at least two groups to finish.
        asm volatile("cp.async.wait_group %0; \n" ::"n"(2));
        __syncthreads();
        // load SA4, SB4
        if (ko + 3 < K / TK)
        {
            loadSmemA(SA4, dA, M, K, ko + 3);
            loadSmemB(SB4, dB, K, N, ko + 3);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA1, SB1
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA1, ki);
            loadFragB(fragB, SB1, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        // load SA1, SB1
        if (ko + 4 < K / TK)
        {
            loadSmemA(SA1, dA, M, K, ko + 4);
            loadSmemB(SB1, dB, K, N, ko + 4);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA2, SB2
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA2, ki);
            loadFragB(fragB, SB2, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        // load SA2, SB2
        if (ko + 5 < K / TK)
        {
            loadSmemA(SA2, dA, M, K, ko + 5);
            loadSmemB(SB2, dB, K, N, ko + 5);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA3, SB3
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA3, ki);
            loadFragB(fragB, SB3, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        // load SA3, SB3
        if (ko + 6 < K / TK)
        {
            loadSmemA(SA3, dA, M, K, ko + 6);
            loadSmemB(SB3, dB, K, N, ko + 6);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA4, SB4
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA4, ki);
            loadFragB(fragB, SB4, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
    }

    // epilogue
    // [K / TK / 4 * 4 - 1 ... K / TK]
    {
        int ko = (K / TK / 4 * 4 - 4);
        // wait for at least two groups to finish.
        asm volatile("cp.async.wait_group %0; \n" ::"n"(2));
        __syncthreads();
        // load SA4, SB4
        if (ko + 3 < K / TK)
        {
            loadSmemA(SA4, dA, M, K, ko + 3);
            loadSmemB(SB4, dB, K, N, ko + 3);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA1, SB1
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA1, ki);
            loadFragB(fragB, SB1, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        // load SA1, SB1
        if (ko + 4 < K / TK)
        {
            loadSmemA(SA1, dA, M, K, ko + 4);
            loadSmemB(SB1, dB, K, N, ko + 4);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA2, SB2
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA2, ki);
            loadFragB(fragB, SB2, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        // load SA2, SB2
        if (ko + 5 < K / TK)
        {
            loadSmemA(SA2, dA, M, K, ko + 5);
            loadSmemB(SB2, dB, K, N, ko + 5);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA3, SB3
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA3, ki);
            loadFragB(fragB, SB3, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        // load SA3, SB3
        if (ko + 6 < K / TK)
        {
            loadSmemA(SA3, dA, M, K, ko + 6);
            loadSmemB(SB3, dB, K, N, ko + 6);
            asm volatile("cp.async.commit_group; \n" ::);
        }
        // compute SA4, SB4
        for (int ki = 0; ki < TK / wmmaK; ki++)
        {
            //warp: TM(64x) x TN(64x) x 16
            loadFragA(fragA, SA4, ki);
            loadFragB(fragB, SB4, ki);
            for (int mi = 0; mi < TM / wmmaM; mi++)
            {
                for (int ni = 0; ni < TN / wmmaN; ni++)
                {
                    //mma: 16 x 16 x 16
                    nvcuda::wmma::mma_sync(Accum[mi * TN / wmmaN + ni], fragA[mi], fragB[ni], Accum[mi * TN / wmmaN + ni]);
                }
            }
        }
    }

    storeAccum(Accum, SmemC);
    __syncthreads();
    storeSmemC(SmemC, dC, M, N);
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

    cudaFuncSetAttribute(gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    dim3 grid(N / BLOCK_TILE_N, M/ BLOCK_TILE_M, 1);
    dim3 block(32, 2, 2);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            o_ptr,
            M, N, K
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
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

    cudaFuncSetAttribute(gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    dim3 grid((N * RATIO) / BLOCK_TILE_N, M / BLOCK_TILE_M, 1);
    dim3 block(32, 2, 2);
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
        gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
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
        gemm_kernel<BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N, BLOCK_TILE_K><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
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

