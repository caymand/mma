#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
mmm_kernel_tt(ProblemShape shape_MNK, CtaTiler cta_tiler,
		   TA const* A, AStride strideA, ASmemLayout sA_layout, AThreadLayout tA,
		   TB const* B, BStride strideB, BSmemLayout sB_layout, BThreadLayout tB,
		   TC      * C, CStride strideC, CSmemLayout sC_layout, CThreadLayout tC,
		   Alpha alpha, Beta beta)
{
/* ASSERTION FOR LAYOUT */
	// Problem has 3 modes
	CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
	CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), strideA)); // (M, K)
	CUTE_STATIC_ASSERT_V(congruent(select<2,1>(shape_MNK), strideB)); // (K, N)
	CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), strideC)); // (M, N)

/* INITIALIZE MATRICES TO TENSORS */
	Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), strideA); // (M, K)	
	Tensor mB = make_tensor(make_gmem_ptr(B), select<2,1>(shape_MNK), strideB); // (K, N)
	Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), strideC); // (M, N)
	
/* TILE THE INPUT MATRICES */
	auto cta_cord = make_coord(blockIdx.x, blockIdx.y, _); // (BLK_M, BLK_N, k)
	Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0,2>(cta_cord)); // (BLK_M, BLK_K, k)
	Tensor gB = local_tile(mB, select<2, 1>(cta_tiler), select<2, 1>(cta_cord)); // (BLK_K, BLK_N, k)
	Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_cord)); // (BLK_M, BLK_N)

/* SHARED MEMORY */
	static_assert(is_static<ASmemLayout>::value);
	static_assert(is_static<BSmemLayout>::value);
	static_assert(is_static<CSmemLayout>::value);
	
	CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
	CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  
	CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<2>(cta_tiler));
	CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<1>(cta_tiler));
	// CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
	// CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

	static_assert(is_static<AThreadLayout>::value);
	static_assert(is_static<BThreadLayout>::value);

	CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
	// Assert that there are no "left over" elements to be copied
	CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});
	CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});
	CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tB) == Int<0>{});
	CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<0>(tB) == Int<0>{});


	__shared__ TA smemA[cosize_v<ASmemLayout>]; // (BLK_M, BLK_K)
	__shared__ TB smemB[cosize_v<BSmemLayout>]; // (BLK_K, BLK_N)

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
	Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_K, BLK_N)

	// Each thread has a 4x1 tile
	Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THX_K, k)
	Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THX_K)	
	// Each thread has a 1x4 tile
	Tensor tBgB = local_partition(gB, tB, threadIdx.x); 
	Tensor tBsB = local_partition(sB, tB, threadIdx.x);
	if (thread0())
	{
		print(tAgA);
		print(tBgB);
	}

/* CALCULATE OUTPUT RESULT TILES*/
	static_assert(is_static<CThreadLayout>::value);
	// Assert output tile for threads is as large as number of threads.
	CUTE_STATIC_ASSERT_V(size(tC) == size(tA));
	CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
	CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});
	
	// Find the data from A & B needed for the 8x8 result for C
	Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M, BLK_K)
	Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N, BLK_K)
	Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{}); // (THR_M, THR_N)	
	// Register accumulator
	Tensor tCrC = make_tensor_like(tCgC);
	
	CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
	CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
	CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
	CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
	CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<0>(tCsB)); // BLK_K

/* MAIN LOOP */
	// Max number of tiles to be done before epiloge. This is the k. 	
	auto K_TILE_MAX = size<2>(tAgA);
	for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
	{
		copy(tAgA(_, _, k_tile), tAsA);
		copy(tBgB(_, _, k_tile), tBsB);

		cp_async_fence();
		cp_async_wait<0>();
		__syncthreads();
		// CUTE_UNROLL
		// for (int k = 0; k < size<1>(tCsA); ++k) {
		// 	CUTE_UNROLL
		// 	for (int m = 0; m < size<0>(tCrC); ++m) {
		// 		CUTE_UNROLL
		// 		for (int n = 0; n < size<1>(tCrC); ++n) {
		// 			tCrC(m,n) += tCsA(m,k) * tCsB(k,n);
		// 		}
		// 	}
		// }
		gemm(tCsA, tCsB, tCrC);
		__syncthreads();
	}
	// Each thread writes back their result to global memory
	axpby(alpha, tCrC, beta, tCgC);
}


template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
mmm_kernel_tn(ProblemShape shape_MNK, CtaTiler cta_tiler,
		   TA const* A, AStride strideA, ASmemLayout sA_layout, AThreadLayout tA,
		   TB const* B, BStride strideB, BSmemLayout sB_layout, BThreadLayout tB,
		   TC      * C, CStride strideC, CSmemLayout sC_layout, CThreadLayout tC,
		   Alpha alpha, Beta beta)
{
/* ASSERTION FOR LAYOUT */
	// Problem has 3 modes
	CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
	// There is a stride defined for each mode of the input matrices.
	CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), strideA)); // (M, K)
	CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), strideB)); // (N, K)
	CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), strideC)); // (M, N)

/* INITIALIZE MATRICES TO TENSORS */
	// The tensor is semantically the tuple of a data pointer and a Layout.
	// The layout is the shape and the stride information.
	Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), strideA); // (M, K)
	Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), strideB); // (N, K)
	Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), strideC); //
	
/* TILE THE INPUT MATRICES */
	// Find the tile for this threadBlock.
	// We simply use block coordinates for this and expect the kernel launch
	// to have created a proper grid. The K dimension is the cute::Underscore type
	// and is thus unspecified. This is becuase this is the sequential reduce dim.
	auto cta_cord = make_coord(blockIdx.x, blockIdx.y, _);

	// Next we have to get a "view" or slice of the input matrices for the block
	// in global memory.
	// For this the local_tile is used. It is a cute inner_partition.
	// This means the inner mode of the tiler is preserved after a zipped_divide.
	// This way, when slicing out the tensor using the tiler, we get the
	// tile size of elements. The outer mode would give the number of tiles.
	// The coord selects the actual tile (this way we get the inner mode).
	// See: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md#cta-partitioning

	// We select only the relevant modes of the tiler and coordinate.
	// The third index ranges over all k tiles. We need to iterate over this
	Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0,2>(cta_cord)); // (BLK_M, BLK_K, k)
	Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_cord)); // (BLK_N, BLK_K, k). TODO: Completely understand this
	Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_cord)); // (BLK_M, BLK_N)

/* SHARED MEMORY */
	// Need shared memory to be statically known
	static_assert(is_static<ASmemLayout>::value);
	static_assert(is_static<BSmemLayout>::value);
	static_assert(is_static<CSmemLayout>::value);
	
	CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
	CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  
	CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
	CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
	// CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
	// CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

	static_assert(is_static<AThreadLayout>::value);
	static_assert(is_static<BThreadLayout>::value);

	// Want the two layouts of threads for copying to match. This wayy all threads
	// can be used to first copy a slice of A and the of B
	CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
	// Assert that there are no "left over" elements to be copied
	CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
	CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
	CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
	CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K


	// Define the shared memory layout. This computes the size of the layout in
	// flat dimensions. cosize is the physical length, so we get it in bytes.
	__shared__ TA smemA[cosize_v<ASmemLayout>]; // (BLK_M, BLK_K)
	__shared__ TB smemB[cosize_v<BSmemLayout>]; // (BLK_N, BLK_K)

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
	Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)

	// Similar to local_tile in that it corresponds to first doing a zipped divide
	// over the tile in global using the layout of threads. However, after
	// the zipped divide, the threadIdx.x is used to index into the first mode,
	// the tile mode.
	// The result has the size of the rest (outer) mode of the zipped dive.
	// This correspponds to how many elements each thread needs to copy.
	// Thus, this gives us a small view of the elements needed to copy to and from.
	Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THX_K, k)
	Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THX_K)
	
	Tensor tBgB = local_partition(gB, tB, threadIdx.x);
	Tensor tBsB = local_partition(sB, tB, threadIdx.x);

/* CALCULATE OUTPUT RESULT TILES*/
	static_assert(is_static<CThreadLayout>::value);
	// Assert output tile for threads is as large as number of threads.
	CUTE_STATIC_ASSERT_V(size(tC) == size(tA));
	CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});
	CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});
	
	// This will transform into -> (BLK_M / THR_M, BLK_K) since we fix the K-mode
	Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1,X>{}); // (THR_M, BLK_K)
	Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N, BLK_K)
	Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{}); // (THR_M, THR_N)	
	// Register accumulator
	Tensor tCrC = make_tensor_like(tCgC);

	CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
	CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
	CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
	CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
	CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLK_K

/* MAIN LOOP */
	// Max number of tiles to be done before epiloge. This is the k 
	auto K_TILE_MAX = size<2>(tAgA);
	for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
	{
		copy(tAgA(_, _, k_tile), tAsA);
		copy(tBgB(_, _, k_tile), tBsB);

		cp_async_fence();
		cp_async_wait<0>();
		__syncthreads();
		    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
		CUTE_UNROLL
		for (int k = 0; k < size<1>(tCsA); ++k) {
			CUTE_UNROLL
			for (int m = 0; m < size<0>(tCrC); ++m) {
			CUTE_UNROLL
				for (int n = 0; n < size<1>(tCrC); ++n) {
					tCrC(m,n) += tCsA(m,k) * tCsB(k,n);
				}
			}
		}
		// gemm(tCsA, tCsB, tCrC);
		__syncthreads();
	}
	// Each thread writes back their result to global memory
	axpby(alpha, tCrC, beta, tCgC);
}
