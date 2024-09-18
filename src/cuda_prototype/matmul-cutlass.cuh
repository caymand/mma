#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>



// Below is opied from cutlass/examples/cute/tutorial/sgemm_1.cu
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyA,
        class TB, class BStride, class BSmemLayout, class TiledCopyB,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_tiledMMA(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride strideGlobalA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride strideGlobalB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride strideGlobalC, CSmemLayout          , TiledMma tiled_mma,
            Alpha alpha, Beta beta)
{
        using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(copy_b));           // NumThreads
    CUTE_STATIC_ASSERT_V(size(tiled_mma) == size(copy_a));                          // NumThreads

    // NO MORE ASSERTIONS ABOUT Thread Layout and tiler. This is handled by the type

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), strideGlobalA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), strideGlobalB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), strideGlobalC));         // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), strideGlobalA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), strideGlobalB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), strideGlobalC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)
    
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);

    // Similar to above. Partition the source data based on the threadIdx.
    // This will be a 8x1 or 4x1 slice depending on the data type. 
    // in the first mode, and we then have a bunch of these slices.
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_Y, CPY_K, k)
    Tensor tArA = make_fragment_like(tAsA);
    
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPU_K, k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPU_K, k)
    Tensor tBrB = make_fragment_like(tBsB);
    
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    // Slice of shared need for MMA and slice of global for result.
    // IN this case, MMA is just 1 element. Not  using tensor cores yet.
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K, K)
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

    // Clear the accumulators
    clear(tCrC);
    // NOTE: tAgA is 4D because we have (CPY, CPY_M, CPY_K, k);
    auto K_TILE_MAX = size<3>(tAgA);
    copy(copy_a, tAgA(_, _,_, 0), tArA);        
    copy(copy_b, tBgB(_, _,_, 0), tBrB);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        __syncthreads();                
        // Copy global->registers for next iteration for next iteration        
        copy(tArA, tAsA);        
        copy(tBrB, tBsB);                
        __syncthreads();         // Wait for all threads to write to smem
        // Initiate copy for next operation
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _,_, k_tile_next), tArA);        
        copy(copy_b, tBgB(_, _,_, k_tile_next), tBrB);

        // Compute gemm on tC thread-partitioned smem
        gemm(tiled_mma, tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
        
    }
    axpby(alpha, tCrC, beta, tCgC);
}



template <class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class AThreadLayout,
        class TB, class BStride, class BSmemLayout, class BThreadLayout,
        class TC, class CStride, class CSmemLayout, class CThreadLayout,
        class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride strideGlobalA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride strideGlobalB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride strideGlobalC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);
    static_assert(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads
    CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), strideGlobalA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), strideGlobalB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), strideGlobalC));         // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), strideGlobalA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), strideGlobalB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), strideGlobalC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    // if (thread0())
    // {
    //     printf("A: %d, B: %d\n", cosize_v<ASmemLayout>, cosize_v<BSmemLayout>);
    // }
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));                // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // THR_K
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // THR_K

    //
    // Define A/B partitioning and C accumulators
    //

    // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

    // Partition sA (M,K) by the rows of tC
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

    // Allocate the accumulators -- same shape/layout as the partitioned data
    Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K

    // Clear the accumulators
    clear(tCrC);

#if 0
    if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
    if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
    if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

    // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
    //           and then computes on those tiles.
    //   copy(.) operates on the global and shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared and register memory via the tC partitioning

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Copy gmem to smem with tA|tB thread-partitioned tensors
        copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
        copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

        // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
        //   Tensor tAgAk = tAgA(_,_,k_tile);
        //   CUTE_UNROLL
        //   for (int i = 0; i < size(tAsA); ++i) {
        //     tAsA(i) = tAgAk(i);
        //   }

        cp_async_fence();        // Label the end of (potential) cp.async instructions
        cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
        __syncthreads();         // Wait for all threads to write to smem

        // Compute gemm on tC thread-partitioned smem
        gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

        // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
        //   CUTE_UNROLL
        //   for (int k = 0; k < size<1>(tCsA); ++k) {
        //     CUTE_UNROLL
        //     for (int m = 0; m < size<0>(tCrC); ++m) {
        //       CUTE_UNROLL
        //       for (int n = 0; n < size<1>(tCrC); ++n) {
        //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
        //       }
        //     }
        //   }

        __syncthreads();         // Wait for all threads to read from smem
    }

#endif

    //
    // Epilogue
    //

//    TODO: remove alpha and beta?
    axpby(alpha, tCrC, beta, tCgC);

    // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent to
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tCsA); ++i) {
    //     tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
    //   }
}
