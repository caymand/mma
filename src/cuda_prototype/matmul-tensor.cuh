//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH


//#define KEEP_C
//#define CACHE_C

#define WARP_SIZE 32

#ifndef SHARED_PADDING
#define SHARED_PADDING 8
#endif


#ifndef LOAD_TYPE
#define LOAD_TYPE float4
#endif

#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif

#ifdef SYNC_CPY
#define USE_PIPELINE
#endif


#include <cstdint>
#include <mma.h>
#include "cuda_fp16.h"

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda;

namespace cg = cooperative_groups;


#define DIV_UP(a, b) (((a) + (b) - 1) / (b))


// TODO: avoid reinterpret_cast<uint32_t *> for below functions

// TODO: account for different elm and acc types
// TODO: check types

__forceinline__ __device__ void ldmatrix_x2(uint32_t r[2], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x2_trans(uint32_t r[2], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x4(uint32_t r[4], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x4_trans(uint32_t r[4], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_ptr));
}

__forceinline__ __device__ void mma_m16n8k16(uint32_t d[4], uint32_t a[4], uint32_t b[2], uint32_t c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}


// TODO: use something like this maybe 2D b and c, else just double dimensions?
//__forceinline__ __device__ void mma_m16n16k16(uint32_t d[4], uint32_t a[4], uint32_t b[2], uint32_t c[4]) {
//    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
//    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
//}


template <unsigned int load_size>
__forceinline__ __device__ void cp_async(void * dst, void * src) {
    auto dst_p = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
//    auto src_p = static_cast<uint32_t>(__cvta_generic_to_global(src));
// TODO: use ignore-src
    if constexpr (load_size == 16) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :  : "r"(dst_p), "l"(src));
    } else if constexpr (load_size == 4) {
//        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" :  : "r"(dst_p), "l"(src));
//        asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], 4;\n" :  : "r"(dst_p), "l"(src));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :  : "r"(dst_p), "l"(src));
    }
}

__forceinline__ __device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :  : );
}

template <int N>
__forceinline__ __device__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :  : "n"(N));
}

__forceinline__ __device__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" :  : );
}

#ifdef USE_PIPELINE
template <class elmType, unsigned int load_size, unsigned int threads_per_block, unsigned int width, unsigned int height, unsigned int core_matrix_width_elms, unsigned int load_tile_width_elms, unsigned int load_tile_height, unsigned int shared_ldm, cuda::thread_scope thread_scope>
__forceinline__ __device__ void copy_global_to_shared_swizzled(elmType * shared, elmType * global, const unsigned int global_offset_x, const unsigned int global_offset_y, const unsigned int global_width, const unsigned int global_height, LOAD_TYPE * zero_elm, cuda::pipeline<thread_scope> &pipeline) {
#else
template <class elmType, unsigned int load_size, unsigned int threads_per_block, unsigned int width, unsigned int height, unsigned int core_matrix_width_elms, unsigned int load_tile_width_elms, unsigned int load_tile_height, unsigned int shared_ldm>
__forceinline__ __device__ void copy_global_to_shared_swizzled(elmType * shared, elmType * global, const unsigned int global_offset_x, const unsigned int global_offset_y, const unsigned int global_width, const unsigned int global_height, LOAD_TYPE * zero_elm) {
#endif
    auto aligned_size = cuda::aligned_size_t<load_size>(load_size);

    constexpr int elms_per_load = DIV_UP(sizeof(LOAD_TYPE), sizeof(elmType));

    constexpr int loads_per_thread = DIV_UP(width * height, elms_per_load * threads_per_block);

    constexpr unsigned int loads_per_core_matrix_row = DIV_UP(core_matrix_width_elms, elms_per_load);

    constexpr unsigned int width_core_matrix_rows = DIV_UP(width, core_matrix_width_elms);

//    TODO: extract
    constexpr unsigned int load_tile_width_core_matrix_rows = DIV_UP(load_tile_width_elms, core_matrix_width_elms);

    #ifdef NOUNROLL2
    #pragma unroll 1
    #else
    #ifdef UNROLL
    #pragma unroll
    #endif
    #endif
    for (int i = 0; i < loads_per_thread; i++)
    {
        unsigned int load_i = i * threads_per_block + threadIdx.x;

        // Consecutive threads load same matrix row
        unsigned int core_matrix_row_i = load_i / loads_per_core_matrix_row;
        unsigned int thread_i_in_core_matrix_row = load_i % loads_per_core_matrix_row;

        unsigned int core_matrix_row_x = core_matrix_row_i % width_core_matrix_rows;
        unsigned int core_matrix_row_y = core_matrix_row_i / width_core_matrix_rows;




//        TODO: extract function?
        unsigned int load_tile_x = core_matrix_row_x / load_tile_width_core_matrix_rows;
        unsigned int load_tile_y = core_matrix_row_y / load_tile_height;

        unsigned int core_matrix_row_x_in_tile = core_matrix_row_x % load_tile_width_core_matrix_rows;
        unsigned int core_matrix_row_y_in_tile = core_matrix_row_y % load_tile_height;

        core_matrix_row_y_in_tile = core_matrix_row_y_in_tile / 2 + (core_matrix_row_y_in_tile % 2) * (load_tile_height / 2);

        #ifdef SWIZZLE
        unsigned int core_matrix_row_y_in_tile_swizzled = core_matrix_row_x_in_tile;
        unsigned int core_matrix_row_x_in_tile_swizzled = core_matrix_row_y_in_tile ^ core_matrix_row_x_in_tile;

        unsigned int core_matrix_row_shared_offset_x = load_tile_x * load_tile_height * core_matrix_width_elms + core_matrix_row_x_in_tile_swizzled * core_matrix_width_elms;
        unsigned int core_matrix_row_shared_offset_y = load_tile_y * load_tile_width_core_matrix_rows + core_matrix_row_y_in_tile_swizzled;
        #else
        unsigned int core_matrix_row_shared_offset_x = load_tile_x * load_tile_width_elms + core_matrix_row_x_in_tile * core_matrix_width_elms;
        unsigned int core_matrix_row_shared_offset_y = load_tile_y * load_tile_height + core_matrix_row_y_in_tile;
        #endif

        auto shared_core_matrix_row_ptr = &shared[core_matrix_row_shared_offset_y * shared_ldm + core_matrix_row_shared_offset_x];

        unsigned int core_matrix_row_global_offset_x = global_offset_x + load_tile_x * load_tile_width_elms + core_matrix_row_x_in_tile * core_matrix_width_elms;
        unsigned int core_matrix_row_global_offset_y = global_offset_y + load_tile_y * load_tile_height + core_matrix_row_y_in_tile;

        auto global_core_matrix_row_ptr = &global[core_matrix_row_global_offset_y * global_width + core_matrix_row_global_offset_x];


        if (load_tile_y < height / load_tile_height)
        {
//            TODO: move out of loop?
            #ifdef EARLY_COMMIT
            #ifdef USE_PIPELINE
            pipeline.producer_acquire();
            #endif
            #endif
            if (core_matrix_row_global_offset_x < global_width && core_matrix_row_global_offset_y < global_height) {
                #ifdef SYNC_CPY
                reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row] = reinterpret_cast<LOAD_TYPE *>(global_core_matrix_row_ptr)[thread_i_in_core_matrix_row];
                #else
                #ifdef USE_PIPELINE
                cuda::memcpy_async(&reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row], &reinterpret_cast<LOAD_TYPE *>(global_core_matrix_row_ptr)[thread_i_in_core_matrix_row], load_size, pipeline);
                #else
                cp_async<load_size>(&reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row], &reinterpret_cast<LOAD_TYPE *>(global_core_matrix_row_ptr)[thread_i_in_core_matrix_row]);
                #endif
                #endif
            } else {
                // TODO: handle zeros, use ignore-src or src-size

                #ifdef SYNC_CPY
                reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row] = LOAD_TYPE();
                #else
                #ifdef USE_PIPELINE
                cuda::memcpy_async(&reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row], &zero_elm, load_size, pipeline);
                #else
                cp_async<load_size>(&reinterpret_cast<LOAD_TYPE *>(shared_core_matrix_row_ptr)[thread_i_in_core_matrix_row], zero_elm);
                #endif
                #endif
            }
            #ifdef EARLY_COMMIT
            #ifdef USE_PIPELINE
            pipeline.producer_commit();
            #else
//            __syncwarp();
            cp_async_commit();
            #endif
            #endif
        }
    }
}


template <bool transpose, class elmType, unsigned int shared_ldm, unsigned int core_matrix_width_elms, unsigned int load_tile_width_elms, unsigned int load_tile_height>
__forceinline__ __device__ void load_frags(unsigned int warpQuarter, unsigned int warpIDInQuarter, uint32_t registers[4], elmType * shared, unsigned int matrix_x, unsigned int matrix_y) {
//  TODO: extract or remove
    constexpr unsigned int load_tile_width_core_matrix_rows = DIV_UP(load_tile_width_elms, core_matrix_width_elms);
    constexpr unsigned int elms_in128B = 128 / sizeof(elmType);



//        TODO: extract function?
    //    TODO: exchange height and width in shared
    unsigned int load_tile_x = matrix_x / load_tile_width_elms;
    unsigned int load_tile_y = matrix_y / load_tile_height + (warpQuarter & 1);

    //    TODO: check this
    unsigned int core_matrix_row_x_in_tile = (matrix_x / core_matrix_width_elms) % load_tile_width_core_matrix_rows + (warpQuarter / 2);
    unsigned int core_matrix_row_y_in_tile = warpIDInQuarter;

    #ifdef SWIZZLE
    unsigned int core_matrix_row_y_in_tile_swizzled = core_matrix_row_x_in_tile;
    unsigned int core_matrix_row_x_in_tile_swizzled = core_matrix_row_y_in_tile ^ core_matrix_row_y_in_tile_swizzled;

    unsigned int core_matrix_row_shared_offset_x = load_tile_x * load_tile_height * core_matrix_width_elms + core_matrix_row_x_in_tile_swizzled * core_matrix_width_elms;
    unsigned int core_matrix_row_shared_offset_y = load_tile_y * load_tile_width_core_matrix_rows + core_matrix_row_y_in_tile_swizzled;
    #else
    unsigned int core_matrix_row_shared_offset_x = load_tile_x * load_tile_width_elms + core_matrix_row_x_in_tile * core_matrix_width_elms;
    unsigned int core_matrix_row_shared_offset_y = load_tile_y * load_tile_height + core_matrix_row_y_in_tile;
    #endif

    unsigned int shared_index = core_matrix_row_shared_offset_y * shared_ldm + core_matrix_row_shared_offset_x;
    auto shared_core_matrix_row_ptr = &shared[shared_index];

    if constexpr (transpose) {
        ldmatrix_x4_trans(registers, shared_core_matrix_row_ptr);
    } else
    {
        ldmatrix_x4(registers, shared_core_matrix_row_ptr);
    }
}


#ifndef THREADS_PER_BLOCK
#ifdef BLOCK_TILES_M
#ifdef BLOCK_TILES_N
#define THREADS_PER_BLOCK BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE
#else
#define THREADS_PER_BLOCK 0
#endif
#else
#define THREADS_PER_BLOCK 0
#endif
#endif

template <class elmType, class accType, unsigned int wmma_m, unsigned int wmma_n, unsigned int wmma_k, unsigned int frags_m, unsigned int frags_n, unsigned int frags_k, unsigned int warp_tiles_m, unsigned int warp_tiles_n, unsigned int warp_tiles_k, unsigned int block_tiles_m, unsigned int block_tiles_n, unsigned int threads_per_block, unsigned int num_stages>
__global__ void
#ifdef BLOCKS_PER_SM
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
#else
__launch_bounds__(THREADS_PER_BLOCK)
#endif
matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
    extern __shared__ __align__(128) char dynamic_shared[];

    auto load_size = cuda::aligned_size_t<sizeof(LOAD_TYPE)>(sizeof(LOAD_TYPE));

    constexpr unsigned int shared_m = wmma_m * frags_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * frags_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * frags_k * warp_tiles_k;

    unsigned int warpID = threadIdx.x / warpSize;
    unsigned int laneID = threadIdx.x % warpSize;

    unsigned int warpQuarter = laneID / 8;
    unsigned int warpIDInQuarter = laneID % 8;

    unsigned int groupID = laneID / 4;
    unsigned int threadID_in_group = laneID % 4;

    // Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

    unsigned int block_m_global_offset = blockIdx.y * shared_m;
    unsigned int block_n_global_offset = blockIdx.x * shared_n;

    // TODO: make this last index instead for better memory access?
    unsigned int warp_m_shared_offset = warp_m_index * wmma_m * frags_m * warp_tiles_m;
    unsigned int warp_n_shared_offset = warp_n_index * wmma_n * frags_n * warp_tiles_n;

    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_shared_offset;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_shared_offset;

    auto A_shared = reinterpret_cast<elmType *>(dynamic_shared);
    #ifdef SWIZZLE
    auto B_shared = A_shared + num_stages * shared_m * shared_k;
    #else
    auto B_shared = A_shared + num_stages * shared_m * (shared_k + SHARED_PADDING);
    #endif


//   TODO: Extract to macro or template argument?
    constexpr unsigned int core_matrix_width_elms = 8;
    constexpr unsigned int core_matrix_height = 8;

    constexpr unsigned int elms_in128B = 128 / sizeof(elmType);

//    TODO: avoid using --expt-relaxed-constexpr?
    constexpr unsigned int load_tile_width_elms_A = std::min(elms_in128B, shared_k);
    constexpr unsigned int load_tile_width_elms_B = std::min(elms_in128B, shared_n);
//    constexpr unsigned int load_tile_width_elms_A = elms_in128B;
//    constexpr unsigned int load_tile_width_elms_B = elms_in128B;


    constexpr int elms_per_load = DIV_UP(sizeof(LOAD_TYPE), sizeof(elmType));
    constexpr int loads_per_thread_A = DIV_UP(shared_m * shared_k, elms_per_load * threads_per_block);
    constexpr int loads_per_thread_B = DIV_UP(shared_k * shared_n, elms_per_load * threads_per_block);


//    TODO: check cutlass (docs) swizzling
    #ifdef SWIZZLE
    constexpr unsigned int shared_ldm_A = std::max(elms_in128B, shared_k);
    constexpr unsigned int shared_ldm_B = std::max(elms_in128B, shared_n);
    #else
    constexpr unsigned int shared_ldm_A = shared_k + SHARED_PADDING;
    constexpr unsigned int shared_ldm_B = shared_n + SHARED_PADDING;
    #endif

//    TODO: choose
    //  TODO: supply as template argument?
//    constexpr unsigned int load_tile_width_core_matrix_rows_A = DIV_UP(load_tile_width_elms_A, core_matrix_width_elms);
//    constexpr unsigned int load_tile_width_core_matrix_rows_B = DIV_UP(load_tile_width_elms_B, core_matrix_width_elms);
//    constexpr unsigned int load_tile_height_A = load_tile_width_core_matrix_rows_A;
//    constexpr unsigned int load_tile_height_B = load_tile_width_core_matrix_rows_B;
    constexpr unsigned int load_tile_height_A = core_matrix_height;
    constexpr unsigned int load_tile_height_B = core_matrix_height;

//    TODO: choose
//    auto zero_elm = LOAD_TYPE();
    __shared__ LOAD_TYPE zero_elm;
    zero_elm = LOAD_TYPE();


    #ifdef USE_PIPELINE
    cg::thread_block block = cg::this_thread_block();
    // Allocate shared storage for a cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
            cuda::thread_scope_block,
            num_stages
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);
// TODO: set num_stages
//    auto pipeline = cuda::make_pipeline();
    #endif

    // TODO: account for different elm and acc types
    // Using 2 x 16x8x16 as basic building block
    float C_frag[frags_m * warp_tiles_m][frags_n * warp_tiles_n][2][4];

    // Initialize C_frag to zero
    #pragma unroll
    for (int warp_m_offset_i = 0; warp_m_offset_i < frags_m * warp_tiles_m; warp_m_offset_i++)
    {
        #pragma unroll
        for (int warp_n_offset_i = 0; warp_n_offset_i < frags_n * warp_tiles_n; warp_n_offset_i++)
        {
            #pragma unroll
            for (int j = 0; j < 2; j++)
            {
                #pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    C_frag[warp_m_offset_i][warp_n_offset_i][j][i] = float();
                }
            }
        }
    }

    unsigned int k_iterations = DIV_UP(k, shared_k);

    #ifdef NOUNROLL1
    #pragma unroll 1
    #else
    #ifdef UNROLL
    #pragma unroll
    #endif
    #endif
    for (int global_k_offset_i = 0; global_k_offset_i < k_iterations + num_stages - 1; global_k_offset_i++) {
        int global_k_offset = global_k_offset_i * shared_k;

        unsigned int load_buffer = global_k_offset_i % num_stages;
        unsigned int compute_buffer = (global_k_offset_i + 1) % num_stages;

        #ifdef SWIZZLE
        auto load_buffer_A = &A_shared[load_buffer * shared_m * shared_k];
        auto load_buffer_B = &B_shared[load_buffer * shared_k * shared_n];
        auto compute_buffer_A = &A_shared[compute_buffer * shared_m * shared_k];
        auto compute_buffer_B = &B_shared[compute_buffer * shared_k * shared_n];
        #else
        auto load_buffer_A = &A_shared[load_buffer * shared_m * (shared_k + SHARED_PADDING)];
        auto load_buffer_B = &B_shared[load_buffer * shared_k * (shared_n + SHARED_PADDING)];
        auto compute_buffer_A = &A_shared[compute_buffer * shared_m * (shared_k + SHARED_PADDING)];
        auto compute_buffer_B = &B_shared[compute_buffer * shared_k * (shared_n + SHARED_PADDING)];
        #endif

        if (global_k_offset_i < k_iterations)
        {
            // Copy A and B to shared memory (Producer Code)
            #ifdef USE_PIPELINE
            #ifndef EARLY_COMMIT
            pipeline.producer_acquire();
            #endif
            copy_global_to_shared_swizzled<elmType, sizeof(LOAD_TYPE), threads_per_block, shared_k, shared_m, core_matrix_width_elms, load_tile_width_elms_A, load_tile_height_A, shared_ldm_A>(load_buffer_A, A, global_k_offset, block_m_global_offset, k, m, &zero_elm, pipeline);
            copy_global_to_shared_swizzled<elmType, sizeof(LOAD_TYPE), threads_per_block, shared_n, shared_k, core_matrix_width_elms, load_tile_width_elms_B, load_tile_height_B, shared_ldm_B>(load_buffer_B, B, block_n_global_offset, global_k_offset, n, k, &zero_elm, pipeline);
            #else
            copy_global_to_shared_swizzled<elmType, sizeof(LOAD_TYPE), threads_per_block, shared_k, shared_m, core_matrix_width_elms, load_tile_width_elms_A, load_tile_height_A, shared_ldm_A>(load_buffer_A, A, global_k_offset, block_m_global_offset, k, m, &zero_elm);
            copy_global_to_shared_swizzled<elmType, sizeof(LOAD_TYPE), threads_per_block, shared_n, shared_k, core_matrix_width_elms, load_tile_width_elms_B, load_tile_height_B, shared_ldm_B>(load_buffer_B, B, block_n_global_offset, global_k_offset, n, k, &zero_elm);
            #ifndef EARLY_COMMIT
            // __syncwarp();
            cp_async_commit();
            #endif
            #endif

            #ifndef EARLY_COMMIT
            #ifdef USE_PIPELINE
            pipeline.producer_commit();
            #endif
            #endif
        } else {
//            TODO: handle differently, handle more than 2 pipeline stages?
            #ifndef USE_PIPELINE
            cp_async_wait_all();
//            Not needed since syncing below
//            __syncthreads();
            #endif
        }

        if (global_k_offset_i >= num_stages - 1) {
            #ifdef USE_PIPELINE
            pipeline.consumer_wait();
//            __syncthreads();
            #else
//            cp_async_wait_all();
            #ifdef EARLY_COMMIT
//            __syncwarp();
            cp_async_wait_group<(loads_per_thread_A + loads_per_thread_B) * (num_stages - 1)>();
            #else
            cp_async_wait_group<num_stages - 1>();
            #endif
            __syncthreads();
            #endif

            // Do Matrix multiplication (Consumer Code)
            if (warp_m_global_offset < m && warp_n_global_offset < n)
            {
                #ifdef NOUNROLL
                #pragma unroll 1
                #else
                #ifdef UNROLL
                #pragma unroll
                #endif
                #endif
                for (int local_k_offset_i = 0; local_k_offset_i < warp_tiles_k; local_k_offset_i++)
                {
                    int local_k_offset = local_k_offset_i * frags_k * wmma_k;

                    #pragma unroll
                    for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                    {
                        int warp_m_offset = warp_m_offset_i * frags_m * wmma_m;

                        half2 A_frag[frags_m][frags_k][4];

                        #pragma unroll
                        for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                        {
                            #pragma unroll
                            for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                            {
                                unsigned int matrix_k_shared_index = local_k_offset + frag_k_offset_i * wmma_k;
                                unsigned int matrix_m_shared_index = warp_m_shared_offset + warp_m_offset + frag_m_offset_i * wmma_m;

                                load_frags<false, elmType, shared_ldm_A, core_matrix_width_elms, load_tile_width_elms_A, load_tile_height_A>(warpQuarter, warpIDInQuarter, reinterpret_cast<uint32_t *>(A_frag[frag_m_offset_i][frag_k_offset_i]), compute_buffer_A, matrix_k_shared_index, matrix_m_shared_index);
                            }
                        }

                        #pragma unroll
                        for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                        {
                            int warp_n_offset = warp_n_offset_i * frags_n * wmma_n;

                            half2 B_frag[frags_k][frags_n][2][2];

                            #pragma unroll
                            for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                            {
                                #pragma unroll
                                for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                                {
                                    unsigned int matrix_k_shared_index = local_k_offset + frag_k_offset_i * wmma_k;
                                    unsigned int matrix_n_shared_index = warp_n_shared_offset + warp_n_offset + frag_n_offset_i * wmma_n;

                                    load_frags<true, elmType, shared_ldm_B, core_matrix_width_elms, load_tile_width_elms_B, load_tile_height_B>(warpQuarter, warpIDInQuarter, reinterpret_cast<uint32_t *>(B_frag[frag_k_offset_i][frag_n_offset_i]), compute_buffer_B, matrix_n_shared_index, matrix_k_shared_index);
                                }
                            }

                            #pragma unroll
                            for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                            {
                                #pragma unroll
                                for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                                {
                                    #pragma unroll
                                    for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                                    {
                                        #ifdef SERPENTINE
                                        // Serpentine iteration to increase temporal locality and reduce register usage
                                        int frag_n_offset_i_serpentine = (frag_m_offset_i % 2) ? (frags_n - 1 - frag_n_offset_i) : frag_n_offset_i;
                                        #else
                                        // Serpentine off
                                        int frag_n_offset_i_serpentine = frag_n_offset_i;
                                        #endif

                                        #pragma unroll
                                        for (int i = 0; i < 2; i++) {
                                            mma_m16n8k16(reinterpret_cast<uint32_t *>(C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine][i]),
                                                         reinterpret_cast<uint32_t *>(A_frag[frag_m_offset_i][frag_k_offset_i]),
                                                         reinterpret_cast<uint32_t *>(B_frag[frag_k_offset_i][frag_n_offset_i_serpentine][i]),
                                                         reinterpret_cast<uint32_t *>(C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine][i]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #ifdef USE_PIPELINE
            pipeline.consumer_release();
//            __syncthreads();
            #else
            __syncthreads();
            #endif
        }
    }

//    TODO: remove?
    __syncthreads();

    if (warp_m_global_offset < m && warp_n_global_offset < n) {
        #pragma unroll
        for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
        {
            int warp_m_offset = warp_m_offset_i * frags_m * wmma_m;

            #pragma unroll
            for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
            {
                int warp_n_offset = warp_n_offset_i * frags_n * wmma_n;

                #pragma unroll
                for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                {
                    #pragma unroll
                    for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                    {
                        unsigned int m_offset = warp_m_global_offset + warp_m_offset + frag_m_offset_i * wmma_m;
                        unsigned int n_offset = warp_n_global_offset + warp_n_offset + frag_n_offset_i * wmma_n;

                        // TODO: vectorize stores, try storing in shared first, then coalesced store to global
                        // TODO: refactor, rename
                        #pragma unroll
                        for (unsigned int j = 0; j < 2; j++)
                        {
                            #pragma unroll
                            for (unsigned int i = 0; i < 4; i++)
                            {
                                unsigned int row = groupID + 8 * (i / 2);
                                unsigned int col = threadID_in_group * 2 + (i & 1);

                                unsigned int m_index = m_offset + row;
                                unsigned int n_index = n_offset + col + j * 8;

                                if (m_index < m && n_index < n)
                                {
                                    C[m_index * n + n_index] = C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i][j][i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


#endif //CODE_MATMUL_TENSOR_CUH
