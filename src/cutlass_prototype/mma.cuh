#include <cute/tensor.hpp>

using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride strideA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride strideB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride strideC, CSmemLayout sC_layout, CThreadLayout tC)
{
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});          // (M, N, K)
    // Assert that the shape and stride are "congurent": same dimensions.
    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), strideA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), strideB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), strideC)); // dC strides for shape MN
    // Assert that the shared memory layouts can be statically known
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);
    // Assert that the tiler and shared memory has the same layout.
    // That is, all the data to be used for the tile can be copied into shared memory
    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K

    static_assert(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N

    // CuTe layout is the tuple (shape, stride)
    auto layoutA = make_layout(select<0,2>(shape_MNK), strideA);
    auto layoutB = make_layout(select<2,1>(shape_MNK), strideB);
    auto layoutC = make_layout(select<0,1>(shape_MNK), strideC);

    // Tensor are pointors to memory and a "layout" - shape and stride information.
    Tensor gA = make_tensor(make_gmem_ptr(A), layoutA);
    Tensor gB = make_tensor(make_gmem_ptr(B), layoutB);
    Tensor gC = make_tensor(make_gmem_ptr(C), layoutC);

    // Want all tiles of K (reduction dimension) so it is given the wildcard
    auto block_coord = make_coord(blockIdx.x, blockIdx.y, _); 
    /*  local_tile is a composition of zipped_divide and slicing out the 
    remaining tensor.    
    see"../../cutlass/media/docs/cute/03_tensor.md"
    This can be used to give each block the sliced out tile.

    This is known as an "inner-partition". 
    That means we keep the 128x8 tile mode and select one for each block. 
    I.e. the below tensor point to 128x8 elements.
    */
    Tensor g_blockA = local_tile(gA, select<0,2>(cta_tiler), select<0,2>(block_coord));
    Tensor g_blockB = local_tile(gB, select<2,1>(cta_tiler), select<2,1>(block_coord));
    Tensor g_blockC = local_tile(gC, select<0,1>(cta_tiler), select<0,1>(block_coord));

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    /* Another option is local_partition that performs a "outer-partition".
    In this case we keep the "outer or rest mode". After the zipped_divide
    we keep the outer mode. This makes each thread point to an element?
    */
    Tensor threadtileAglobalA = local_partition(gA, tA, threadIdx.x);
    Tensor threadtileAsharedA = local_partition(sA, tA, threadIdx.x);
    Tensor threadtileBglobalB = local_partition(gB, tA, threadIdx.x);
    Tensor threadtileBsharedB = local_partition(sA, tA, threadIdx.x);

    CUTE_STATIC_ASSERT_V(size<0>(threadtileAglobalA) == size<0>(threadtileAsharedA));  // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(threadtileAglobalA) == size<1>(threadtileAsharedA));  // THR_K
    CUTE_STATIC_ASSERT_V(size<0>(threadtileBglobalB) == size<0>(threadtileBsharedB));  // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(threadtileBglobalB) == size<1>(threadtileBsharedB));  // THR_K

    // Parition the threads tile for C 
    // The _ is a cute::Underscore type and its semantics is like [:, x] in python. Retain that mode
    Tensor threadtileCsharedA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});

}