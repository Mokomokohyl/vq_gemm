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

namespace cg = cooperative_groups;

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
#define MAX_SHARED_MEMORY_USAGE (2 * BLOCK_TILE_N * BLOCK_TILE_K * sizeof(half) + (128 / 8 * 256 * 4 * 2))
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

template<uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc(){
  asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
}

template<uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc(){
  asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
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
    // 不使用 ldmatrix：按元素加载每个 16x8 子块的两个 8x8 子片，
    // 再用 movmatrix 在寄存器内做 8x8 转置，满足 mma.sync.row.col 对 B 的列主布局要求。
    // Warp 负责 16x64（分成 8 个 16x8 小块），每次每线程加载 2 组 1x2 half（j=0/1 对应 8 行偏移）。
    #pragma unroll
    for (int i = 0; i < 8; i++) {       // 8 个 16x8 小块横向遍历
        for (int j = 0; j < 2; j++) {   // 将 16 行拆成两个 8x8（j 决定 +0 或 +8 行偏移）
            int row = ki * WARP_TILE_K + j * 8 + (lane_id / 4);
            int col = warp_id_y * WARP_TILE_N + i * 8 + (lane_id % 4) * 2;

            uint32_t val = *(uint32_t*)(
                shmem +
                (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * WMMA_TILE_N) +
                (col / WMMA_TILE_N) * (WMMA_TILE_K * WMMA_TILE_N) +
                (row % WMMA_TILE_K) * WMMA_TILE_N +
                (col % WMMA_TILE_N)
            );

            // 将按行的 1x2 打包数据就地转置到 mma B 片段需要的列主布局
            asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %0;\n" : "+r"(val));
            frag[i * 2 + j] = val;
        }
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
        *(uint32_t*)(&shmem[(threadIdx.x / 64)                  * (8 * 16 * 16) + 
                            (threadIdx.x % 4 * 16 + i) * 2 / 16 * (16 * 16) + 
                            (threadIdx.x / 4)              % 16 * 16 + 
                            (threadIdx.x % 4 * 8 + i) * 2       % 16]) 
        = *(uint32_t*)(&codebook_shmem[(local_id + i / 4) * 256 * RATIO + ((uint32_t) indices[i]) * RATIO]);
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

template<const int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) e2e_gemm_kernel(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K,
    int* _ready,    // global ready flags: 2 per y-group (double buffer), value = generation
    int* _consumed  // global consumed counters: 2 per y-group (cumulative)
)
{
    cg::cluster_group cluster       = cg::this_cluster();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid = threadIdx.x;

    // Determine group index along grid.y to index global lock arrays
    const int group_id = blockIdx.y; // one 2-slot lock per y-group
    int* B_ready[2];
    int* B_consumed[2];
    // Two slots per group for double buffering
    B_ready[0]    = _ready    ? &_ready[group_id * 2 + 0]    : nullptr;
    B_ready[1]    = _ready    ? &_ready[group_id * 2 + 1]    : nullptr;
    B_consumed[0] = _consumed ? &_consumed[group_id * 2 + 0] : nullptr;
    B_consumed[1] = _consumed ? &_consumed[group_id * 2 + 1] : nullptr;
    // Volatile aliases for lock reads (avoid atomicAdd(...,0) for polling)
    volatile int* V_ready[2];
    volatile int* V_consumed[2];
    V_ready[0]    = (volatile int*)B_ready[0];
    V_ready[1]    = (volatile int*)B_ready[1];
    V_consumed[0] = (volatile int*)B_consumed[0];
    V_consumed[1] = (volatile int*)B_consumed[1];
    // Initialize global lock arrays once per y-group inside kernel
    if (cluster_block_id == 0 && tid == 0) {
        if (B_ready[0])    *B_ready[0] = -1;
        if (B_ready[1])    *B_ready[1] = -1;
        if (B_consumed[0]) *B_consumed[0] = 0;
        if (B_consumed[1]) *B_consumed[1] = 0;
    }
    cluster.sync();

    /* block specialization begin */
    if (cluster_block_id == CLUSTER_SIZE - 1) {
// begin: dequant block
    // Load codebook
    extern __shared__ uint8_t shmem[];
    half *B[2];
    B[0] = reinterpret_cast<half*>(shmem);
    B[1] = reinterpret_cast<half*>(shmem + BLOCK_TILE_K * BLOCK_TILE_N * sizeof(half));
    // B_ready / B_consumed already set to global pointers above

    half *codebook_buf = reinterpret_cast<half*>(shmem + (2 * BLOCK_TILE_K * BLOCK_TILE_N) * sizeof(half));
    load_codebook(codebook_buf, _codebook);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    // Publish initial two generations (g=0,1) without waiting
    dequantToShmemB(B[0], _w, _codebook, codebook_buf, K, N, 0);
    if (tid == 0) {
        atomicExch(B_ready[0], 0); // gen = 0 on idx 0
        // printf("[PROD y=%d] publish buf0 gen=0\n", group_id);
    }
    dequantToShmemB(B[1], _w, _codebook, codebook_buf, K, N, 1);
    if (tid == 0) {
        atomicExch(B_ready[1], 1); // gen = 1 on idx 1
        // printf("[PROD y=%d] publish buf1 gen=1\n", group_id);
    }
    __syncthreads();

    for (int ko = 2; ko < K / BLOCK_TILE_K; ko++) {
        // Double buffer index for this ko
        int idx = ko & 1;
        // Before reusing buffer idx, producer waits for cumulative consumption of previous reuse
        int target_prev = (ko / 2) * (CLUSTER_SIZE - 1);
        if (tid == 0) {
            while ((*V_consumed[idx]) < target_prev) {
                __nanosleep(32);
            }
        }
        __syncthreads();

        // Fill and publish generation ko on buffer idx
        dequantToShmemB(B[idx], _w, _codebook, codebook_buf, K, N, ko);
        if (tid == 0) {
            atomicExch(B_ready[idx], ko); // publish exact generation
        }
    }
// end:   dequant block
    } else {
// begin: GeMM blcoks
    extern __shared__ uint8_t shmem[];
    half *A1 = reinterpret_cast<half*>(shmem); // BLOCK_TILE_M * BLOCK_TILE_K
    // get remote addresses in dequant block
    // B Buffer
    half* B[2];
    B[0] = reinterpret_cast<half*>(cluster.map_shared_rank(shmem, CLUSTER_SIZE - 1));
    B[1] = reinterpret_cast<half*>(cluster.map_shared_rank(shmem + BLOCK_TILE_K * BLOCK_TILE_N * sizeof(half), CLUSTER_SIZE - 1));
    // B_ready / B_consumed already set to global pointers above

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        loadShmemA(A1, _input, M, K, ko);
        asm volatile("cp.async.wait_all;\n"::);
        // Wait for exact generation on buffer idx
        if (tid == 0) {
            while ((*V_ready[ko % 2]) != ko) { __nanosleep(32); }
        }
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A1, ki);
            loadFragB_mma(B_frags, B[ko % 2], ki);
            // dequantToRegB(B_frags, _w, _codebook, codebook_buf, K, N, ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
        if (tid == 0) { atomicAdd(B_consumed[ko % 2], 1); } // signal consumption
        __syncthreads();
    }
    storeC(_o, C_frags, M, N * RATIO);    
// end:   GeMM block
    }

    cluster.sync();
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

    const int CLUSTER_SIZE = 16;
    printf("CLUSTER_SIZE: %d, grid:(%d, %d)", CLUSTER_SIZE, CLUSTER_SIZE, N * RATIO / BLOCK_TILE_N);
    dim3 grid(CLUSTER_SIZE, N * RATIO / BLOCK_TILE_N);
    dim3 block(BLOCK_SIZE);
    cudaFuncSetAttribute(e2e_gemm_kernel<CLUSTER_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    cudaFuncSetAttribute(e2e_gemm_kernel<CLUSTER_SIZE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    // Allocate global lock arrays: 2 slots per y-group (double buffering)
    int groups = grid.y;
    int lock_slots = groups * 2;
    int *d_ready = nullptr, *d_consumed = nullptr;
    cudaMalloc(&d_ready,    lock_slots * sizeof(int));
    cudaMalloc(&d_consumed, lock_slots * sizeof(int));
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        e2e_gemm_kernel<CLUSTER_SIZE><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            o_ptr,
            M, N, K,
            d_ready, d_consumed
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        e2e_gemm_kernel<CLUSTER_SIZE><<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            o_ptr,
            M, N, K,
            d_ready, d_consumed
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
    // Optionally, could copy back errors for debugging; omitted to keep interface clean
    cudaFree(d_ready);
    cudaFree(d_consumed);
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
    std::cout << "TFLOPS : " << ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    return o;
}
