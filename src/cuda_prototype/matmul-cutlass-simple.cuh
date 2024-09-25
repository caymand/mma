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


template <class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyAGlobalShared, class TiledCopyASharedRegisters,
        class TB, class BStride, class BSmemLayout, class TiledCopyBGlobalShared, class TiledCopyBSharedRegisters,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_simple(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyBGlobalShared copyB_global_shared, TiledCopyBSharedRegisters tiled_copy_shared_regs_B,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyAGlobalShared copyA_global_shared, TiledCopyASharedRegisters tiled_copy_shared_regs_A, 
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
Alpha alpha, Beta beta)
{
    using namespace cute;
#if 1
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    // TODO: Should these be used?
    // CUTE_STATIC_ASSERT_V(size(copyA_global_shared) == size(mma));                     // NumThreads
    // CUTE_STATIC_ASSERT_V(size(copyA_global_shared) == size(mma));                     // NumThreads

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
    
    ThrCopy thr_copy_a_global_shared = copyA_global_shared.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a_global_shared.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a_global_shared.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA);                                            // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b_global_shared = copyB_global_shared.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b_global_shared.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b_global_shared.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                                            // (CPY,CPY_N,CPY_K)
    
    // Prefetch into registers  
    copy(copyA_global_shared, tAgA(_,_,_,0), tArA);
    copy(copyB_global_shared, tBgB(_,_,_,0), tBrB);
    
#if 1
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K
#endif 
    // auto smem_thr_copy_A = tiled_copy_shared_regs_A.get_thread_slice(threadIdx.x);
    // Tensor tCrC = smem_thr_copy_A.partition_S(sA);
        
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);

    Tensor tCsA = thr_mma.partition_A(sA);                              // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)        
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
#if 1
    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K   
#endif
    // Clear the accumulators
    clear(tCrC);
  
    int k_tile_max = size<3>(tAgA);
    
    for (int k_tile = 0; k_tile < k_tile_max; k_tile++)
    {
        // Copy into shared, using the prefetch into register from earlier
        __syncthreads();         // Wait for all threads to consume smem
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();         // Wait for all threads to consume smem
        // Prefetch for next iteration (if it is safe to do so)
        int k_tile_next = (k_tile + 1 < k_tile_max) ? k_tile + 1 : k_tile;
        copy(copyA_global_shared, tAgA(_,_,_,k_tile_next), tArA);
        copy(copyB_global_shared, tBgB(_,_,_,k_tile_next), tBrB);
        
        gemm(mma, tCsA, tCsB, tCrC);
    }
    // Write back to global with result
    axpby(alpha, tCrC, beta, tCgC);
}
