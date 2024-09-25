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


template<class SMemALayout,
        class SMemBLayout,
        class SMemCLayout,
        class SmemCopyOpA,
        class SmemCopyOpB,
        class SmemCopyOpC,
        uint32_t ThreadBlockSize,
        class TiledMma,
        uint32_t CopyMaxVecBits,
        class TA,
        class TB,
        class TC,
        class Alpha,
        class Beta,
        class ALayout,
        class BLayout,
        class CLayout
>
__launch_bounds__(ThreadBlockSize) __global__ void
cooperative_gemm_kernel(
    TA const*   A, ALayout a_layout,
    TB const*   B, BLayout b_layout,
    TC*         C, CLayout c_layout,
//    TODO: remove these args?
    TC*         c_out,
    Alpha const alpha,
    Beta  const beta
)
{
    using namespace cute;

    Tensor g_a_tensor     = make_tensor(make_gmem_ptr(A), a_layout);
    Tensor g_b_tensor     = make_tensor(make_gmem_ptr(B), b_layout);
    Tensor g_c_tensor     = make_tensor(make_gmem_ptr(C), c_layout);
    Tensor g_c_out_tensor = make_tensor(make_gmem_ptr(c_out), c_layout);

    constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

    extern __shared__ float4 smem_buf[];
    auto* smem_ptr = reinterpret_cast<unsigned char*>(smem_buf);
    auto* smem_ptr_a = smem_ptr;
    auto* smem_ptr_b = smem_ptr_a + round_up((sizeof(TA) * cosize(SMemALayout{})), copy_max_vec_bytes);
    auto* smem_ptr_c = smem_ptr_b + round_up((sizeof(TB) * cosize(SMemBLayout{})), copy_max_vec_bytes);

    Tensor s_a_tensor = make_tensor(make_smem_ptr<TA>(smem_ptr_a), SMemALayout{});
    Tensor s_b_tensor = make_tensor(make_smem_ptr<TB>(smem_ptr_b), SMemBLayout{});
    Tensor s_c_tensor = make_tensor(make_smem_ptr<TC>(smem_ptr_c), SMemCLayout{});

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_a_tensor, s_a_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_b_tensor, s_b_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_c_tensor, s_c_tensor);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
//    TODO: replace by custom code, avoid c in shared?
    cooperative_gemm<SmemCopyOpA, SmemCopyOpB, SmemCopyOpC>(
            threadIdx.x, tiled_mma,
            alpha, s_a_tensor, s_b_tensor, beta, s_c_tensor
    );
    __syncthreads();

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, s_c_tensor, g_c_out_tensor);
}
