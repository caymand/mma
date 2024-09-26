/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>


template <
        class CopyOpA, class CopyOpB,
        class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyAGlobalShared,
        class TB, class BStride, class BSmemLayout, class TiledCopyBGlobalShared,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_simple(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyAGlobalShared copyA_global_shared,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyBGlobalShared copyB_global_shared,
            TC      * C, CStride dC, CSmemLayout sC_layout, TiledMma tiled_mma,
            Alpha alpha, Beta beta
)
{
    using namespace cute;

#if 1
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copyA_global_shared) == size(tiled_mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copyB_global_shared) == size(tiled_mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
#endif

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));  // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)


//    TODO: use collective copy?
    ThrCopy thr_copy_a_global_shared = copyA_global_shared.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a_global_shared.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a_global_shared.partition_D(sA);                            // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b_global_shared = copyB_global_shared.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b_global_shared.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b_global_shared.partition_D(sB);                            // (CPY,CPY_N,CPY_K)

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)


//    Tensor tCsA = thr_mma.partition_A(sA);
//    Tensor tCsB = thr_mma.partition_B(sB);


    // Create register tensors for the MMA to operate on
    Tensor tCrA  = thr_mma.partition_fragment_A(sA);                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB);                    // (MMA,MMA_N,MMA_K)

    auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<CopyOpA, TA>{}, thr_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));             // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<CopyOpB, TB>{}, thr_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K
    
#if 1
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCrA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCrB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                // MMA_K
#endif

    // Clear the accumulators
    clear(tCrC);


//    TODO: try pipelining this loop
    int k_tile_max = size<3>(tAgA);
    for (int k_tile = 0; k_tile < k_tile_max; k_tile++)
    {
        // Copy global -> shared
        copy(copyA_global_shared, tAgA(_,_,_,k_tile), tAsA);
        copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();


        // Inner loop

        // PREFETCH
        copy(smem_tiled_copy_A, tCsA(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));

        constexpr int K_BLOCK_MAX = size<2>(tCrA);

        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            // static-if load the next k_block. No k-predication required on these loads.
            if (k_block < K_BLOCK_MAX-1)
            {
                // Load the next k_block
                int k_next = k_block + 1;       // statically unrolled
                copy(smem_tiled_copy_A, tCsA(_,_,k_next), tCrA_copy_view(_,_,k_next));
                copy(smem_tiled_copy_B, tCsB(_,_,k_next), tCrB_copy_view(_,_,k_next));
            }

            // GEMM on k_block in registers
            gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
//        gemm(thr_mma, tCsA, tCsB, tCrC);
        __syncthreads();
    }

    // Write back to global with result
    axpby(alpha, tCrC, beta, tCgC);
}
