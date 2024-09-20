#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.cuh"
#include "matmul-tensor-naive.cuh"
#include "matmul-tensor.cuh"
#include "matmul-cutlass.cuh"
//#include "cuda_fp16.h"
#include "cutlass/half.h"
#include <cassert>
//#include <cublas.h>
#include <cublas_v2.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"


#define WARP_SIZE 32
#define SHARED_MEM_SIZE 49152
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_REGISTERS_PER_BLOCK 65536

#ifndef SHARED_PADDING
#define SHARED_PADDING 8
#endif

typedef cutlass::half_t half_t;


enum mm_kernel {
    register_tiled,
    tensor_naive,
    tensor_optimized,
    cublas,
    cutlass_default,
    cutlass_mm
};


template <typename elmT, typename elmAccT = elmT>
long int benchmark_optimized_tensor_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{
// Set constants using compiler options
#ifdef WMMA_M
    constexpr int wmma_m = WMMA_M;
#else
    constexpr int wmma_m = 16;
#endif
#ifdef WMMA_N
    constexpr int wmma_n = WMMA_N;
#else
    constexpr int wmma_n = 16;
#endif
#ifdef WMMA_K
    constexpr int wmma_k = WMMA_K;
#else
    constexpr int wmma_k = 16;
#endif
#ifdef FRAGS_M
    constexpr int frags_m = FRAGS_M;
#else
    constexpr int frags_m = 2;
#endif
#ifdef FRAGS_N
    constexpr int frags_n = FRAGS_N;
#else
    constexpr int frags_n = 2;
#endif
#ifdef FRAGS_K
    constexpr int frags_k = FRAGS_K;
#else
    constexpr int frags_k = 1;
#endif
#ifdef WARP_TILES_M
    constexpr int warp_tiles_m = WARP_TILES_M;
#else
    constexpr int warp_tiles_m = 1;
#endif
#ifdef WARP_TILES_N
    constexpr int warp_tiles_n = WARP_TILES_N;
#else
    constexpr int warp_tiles_n = 1;
#endif
#ifdef WARP_TILES_K
    constexpr int warp_tiles_k = WARP_TILES_K;
#else
    constexpr int warp_tiles_k = 4;
#endif
#ifdef BLOCK_TILES_M
    constexpr int block_tiles_m = BLOCK_TILES_M;
#else
    constexpr int block_tiles_m = 2;
#endif
#ifdef BLOCK_TILES_N
    constexpr int block_tiles_n = BLOCK_TILES_N;
#else
    constexpr int block_tiles_n = 2;
#endif

    constexpr unsigned int threads_per_block = block_tiles_m * block_tiles_n * WARP_SIZE;
    printf("    Threads used: %d/%d\n", threads_per_block, MAX_THREADS_PER_BLOCK);
    assert(threads_per_block <= MAX_THREADS_PER_BLOCK);
    // Assumes num_warps >= block_tiles_m * block_tiles_n, i.e. all block tiles are handled by a warp
    assert(threads_per_block / WARP_SIZE >= block_tiles_m * block_tiles_n);

    printf("    Using wmma %d x %d x %d\n", wmma_m, wmma_n, wmma_k);
    printf("    Using frags %d x %d x %d\n", frags_m, frags_n, frags_k);
    printf("    Using warp tiles %d x %d x %d\n", warp_tiles_m, warp_tiles_n, warp_tiles_k);
    printf("    Using block tiles %d x %d\n", block_tiles_m, block_tiles_n);

    constexpr unsigned int shared_m = wmma_m * frags_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * frags_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * frags_k * warp_tiles_k;

    int dimx = ceil(((float) n)/(shared_n));
    int dimy = ceil(((float) m)/(shared_m));

    dim3 grid(dimx, dimy, 1);
    dim3 block(threads_per_block, 1, 1);

    printf("    Blocks used: %d x %d = %d\n", dimx, dimy, dimx * dimy);

    printf("    Available registers per thread: %d (%d per block)\n", MAX_REGISTERS_PER_BLOCK / threads_per_block, MAX_REGISTERS_PER_BLOCK);

    int max_shared_memory;
    cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    constexpr unsigned int num_stages = NUM_STAGES;

    #ifdef SWIZZLE
    constexpr unsigned int shared_memory_used_A = shared_m * shared_k * sizeof(elmT) * num_stages;
    constexpr unsigned int shared_memory_used_B = shared_k * shared_n * sizeof(elmT) * num_stages;
    #else
    constexpr unsigned int shared_memory_used_A = shared_m * (shared_k + SHARED_PADDING) * sizeof(elmT) * num_stages;
    constexpr unsigned int shared_memory_used_B = shared_k * (shared_n + SHARED_PADDING) * sizeof(elmT) * num_stages;
    #endif

    constexpr unsigned int shared_memory_used = shared_memory_used_A + shared_memory_used_B;

    printf("    Shared memory used: %d/%d bytes (%.0f%%)\n", shared_memory_used, max_shared_memory, (float) shared_memory_used / max_shared_memory * 100);
    printf("    Shared memory used A: %d/%d bytes (%.0f%%)\n", shared_memory_used_A, max_shared_memory, (float) shared_memory_used_A / max_shared_memory * 100);
    printf("    Shared memory used B: %d/%d bytes (%.0f%%)\n", shared_memory_used_B, max_shared_memory, (float) shared_memory_used_B / max_shared_memory * 100);

    auto kernel = matMulTiledTensor<elmT, elmAccT, wmma_m, wmma_n, wmma_k, frags_m, frags_n, frags_k, warp_tiles_m, warp_tiles_n, warp_tiles_k, block_tiles_m, block_tiles_n, threads_per_block, num_stages>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
//    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
//        TODO: fix requested amount of shared memory
        kernel<<<grid, block, shared_memory_used>>>(
            A_device, B_device, C_device, m, n, k
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


//// Setup params for a NT GEMM
//template <class TA, class TB, class TC,
//        class Alpha, class Beta>
//void
//gemm_nt(int m, int n, int k,
//        Alpha alpha,
//        TA const* A, int ldA,
//        TB const* B, int ldB,
//        Beta beta,
//        TC      * C, int ldC,
//        cudaStream_t stream = 0)
//{
// TODO: generalize to any elm types
template <typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_mmm(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

// TODO: base on TN to match tensor cores?
template<>
long int benchmark_cutlass_mmm<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define strides (mixed)
    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, N);                      // (dN, dK)
    auto dC = make_stride(N, Int<1>{});                      // (dM, dN)

    // Define CTA tile sizes (static)
//    TODO: get from calculation
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<16>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<3>{};  // Pipeline

    auto sA_buffer = make_layout(make_shape(bM, bK), make_stride(bK + Int<8>{}, Int<1>{}));
    auto sB_buffer = make_layout(make_shape(bN, bK), make_stride(Int<1>{}, bN + Int<8>{}));

    // Define the smem layouts (static)
//    auto sA = make_layout(make_shape(bM, bK, bP), make_stride(bK, Int<1>{}));
//    auto sB = make_layout(make_shape(bK, bN, bP), LayoutRight{});
    auto sA = tile_to_shape(sA_buffer, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_buffer, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

//    TODO: calculate layouts based on size of elements
    // Define the thread layouts (static)
//    TODO try other cache and Zfill
    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        // TODO: calculate instead
        Layout<Shape<_128, _2>, Stride<_2, _1>>{},
        Layout<Shape<_1, _8>, Stride<_8, _1>>{}
    );
    TiledCopy copyB_global_shared = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _16>, Stride<_16, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _8>>{}
    );

//    TiledMMA mmaC = make_tiled_mma(UniversalFMA<float,half_t,half_t>{},
//        Layout<Shape<_16,_16,_1>>{} // 16x16x1 TiledMMA
//    );

    TiledMMA mmaC = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
//            Layout<Shape<_2,_4,_1>>{}
//            Layout<Shape<_2,_4,_1>, Stride<_4,_1,_8>>{},
//            Tile<_32, _32, _16>{}
//            Layout<Shape<_1,_1>>{},
//            Tile<_32, _32, _16>{}
            Layout<Shape<_2,_4,_1>>{},
            Tile<_128, _128, _16>{}
    );

    TiledCopy copyA_shared_registers = make_tiled_copy_A(Copy_Atom<UniversalCopy<half_t>, half_t>{}, mmaC);
    TiledCopy copyB_shared_registers = make_tiled_copy_B(Copy_Atom<UniversalCopy<half_t>, half_t>{}, mmaC);
//    TODO: handle C in same way

//    print(mmaC);
//    print_latex(mmaC);
//    print_latex(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{});
//    print_latex(copyA_shared_registers);

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        gemm_device<<<dimGrid, dimBlock, 0>>>(
                prob_shape, cta_tiler,
                A, dA, sA, copyA_global_shared, copyA_shared_registers,
                B, dB, sB, copyB_global_shared, copyB_shared_registers,
                C, dC, sC, mmaC,
                Int<1>{}, Int<0>{});
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


// TODO: generalize to any elm types
template <typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_default(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

template<>
long int benchmark_cutlass_default<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using CutlassGemm = cutlass::gemm::device::Gemm<
            half_t,        // Data-type of A matrix
            cutlass::layout::RowMajor,  // Layout of A matrix
            half_t,        // Data-type of B matrix
            cutlass::layout::RowMajor,  // Layout of B matrix
            float,        // Data-type of C matrix
            cutlass::layout::RowMajor,  // Layout of C matrix
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80
    >;

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, k},    // Tensor-ref for source matrix A
                                {B, n},    // Tensor-ref for source matrix B
                                {C, n},    // Tensor-ref for source matrix C
                                {C, n},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {1, 0}); // Scalars used in the Epilogue


    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        gemm_operator(args);
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT = elmT>
unsigned benchmark_naive_tensor_mmm(
        unsigned n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *ResMat_device,
        int m,
        int n,
        int k)
{
    constexpr int block_tiles_m = 8;
    constexpr int block_tiles_n = 4;
    constexpr int block_tiles_k = 4;
    constexpr int wmma_n = 16;
    constexpr int wmma_m = 16;
    constexpr int wmma_k = 16;


    // Let block work on block_tiles * wmma elements.
    // there are n elements on the x direction and we know each thread works on block_tiles_n
    int dimx = ceil(((float) n)/(wmma_n * block_tiles_n));
    int dimy = ceil( ((float) m)/(wmma_m * block_tiles_m));
    dim3 grid(dimx, dimy, 1);
    // dim3 block(threads_per_block, 1, 1); // 1D block of 256 elements
    /* Okay so what do we want? Each mm will be done by the entire warp and works warp level.
    So whatever we want to tile for should be multiple of the warp size.
    Here we say that the block should compute block_tiles_m x block_tiles_n tensor mm.

    This also works for the grid specification, since we tile so that each warp computes
    a wmma_m x wmma_n result, and we use block_tiles_m x block_tiles_n warps in the block.
    */
    dim3 block(block_tiles_n * WARP_SIZE, block_tiles_m, 1);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensorNaive<
            elmAccT, elmT, wmma_m, wmma_n, wmma_k, block_tiles_m, block_tiles_n, block_tiles_k>
            <<<grid, block>>>(A_device, B_device, ResMat_device, m, n, n);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());

    return t.elapsed();
}


template <typename elmT, typename elmAccT>
long int benchmark_tiled_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{
    constexpr int tile_size = 16;
    constexpr int reg_size = 5;

    int dimy = ceil( ((float) n)/(tile_size * reg_size));
    int dimx = ceil( ((float) m)/(tile_size * reg_size));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiled<elmT, elmAccT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                A_device, B_device, C_device, m, n, k);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT>
cublasStatus_t cublas_wrapper(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const elmAccT *alpha,
        const elmT *A, int lda,
        const elmT *B, int ldb,
        const elmAccT *beta,
        elmAccT *C, int ldc
);

template <>
cublasStatus_t cublas_wrapper<half_t, half_t>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half_t *alpha,
        const half_t *A, int lda,
        const half_t *B, int ldb,
        const half_t *beta,
        half_t *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            beta,
            C, CUDA_R_16F, ldc,
            CUDA_R_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

template <>
cublasStatus_t cublas_wrapper<float, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const float *A, int lda,
        const float *B, int ldb,
        const float *beta,
        float *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_32F, lda,
            B, CUDA_R_32F, ldb,
            beta,
            C, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}


template <>
cublasStatus_t cublas_wrapper<half_t, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const half_t *A, int lda,
        const half_t *B, int ldb,
        const float *beta,
        float *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            beta,
            C, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}


template <typename elmT, typename elmAccT>
long int benchmark_cublas(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{
    TimeMeasurement t;

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    elmAccT alpha = (elmAccT) 1.0;
    elmAccT beta = (elmAccT) 0.0;
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    t.start();
    for (int i = 0; i < n_runs; i++) {
        stat = cublas_wrapper<elmT, elmAccT>(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha,
            // Cublas uses column major, so we need to swap A and B, since B^T @ A^T = (A @ B)^T = C^T
            B_device, n,
            A_device, k,
            &beta,
            C_device, n
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS error\n");
        printf("%s\n", cublasGetStatusName(stat));
        printf("%s\n", cublasGetStatusString(stat));
        exit(1);
    }

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename elmAccT, int MatDim, mm_kernel kernel_type>
void run_mmm_kernel(
        int n_runs,
        int m,
        int n,
        int k,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B,
        RandomMatrix<elmAccT, MatDim> &C)
{
    double total_ops = 2.0f * n * k * m;

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();

    auto C_device = C.to_gpu();
    long int total_elapsed;

    if constexpr (kernel_type == mm_kernel::tensor_optimized) {
        total_elapsed = benchmark_optimized_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else if constexpr (kernel_type == mm_kernel::tensor_naive) {
        total_elapsed = benchmark_naive_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else if constexpr (kernel_type == mm_kernel::cublas) {
        total_elapsed = benchmark_cublas<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else if constexpr (kernel_type == mm_kernel::cutlass_mm) {
        total_elapsed = benchmark_cutlass_mmm(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else if constexpr (kernel_type == mm_kernel::cutlass_default) {
        total_elapsed = benchmark_cutlass_default(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else {
        total_elapsed = benchmark_tiled_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }

    cudaMemcpy(C.to_cpu(), C_device, C.flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(C.to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
        printf("Average Time elapsed: %ld ms\n", total_elapsed / n_runs);
    }
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename accT, int MatDim, mm_kernel kernel_type, bool validate>
void benchmark_kernel(
        int n_runs,
        int m,
        int n,
        int k,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B,
        RandomMatrix<accT, MatDim> &C,
        RandomMatrix<accT, MatDim> &C_target,
        std::string kernel_name
    ) {
    C.fill_zeros(m, n);

    std::cout << "-----" << std::endl;
    std::cout << "Running " << kernel_name << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
            1, m, n, k, A, B, C
    );

    RandomMatrix<accT, MatDim> C_actual;

    if constexpr (validate) {
        C_actual.fill_from(C, m, n);
    }

    std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
    run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
            n_runs, m, n, k, A, B, C
    );
    std::cout << "-----" << std::endl;

    if constexpr (validate)
    {
        Validator<accT> validator(C_target.to_cpu(), C_actual.to_cpu(), m * n);
        // validator.setEps(0.000005); // original used by cosmin
        validator.setEps(0.0005);

        validator.validate();
    }
}


#ifdef ELM_T
typedef ELM_T element_type;
#else
typedef half_t element_type;
#endif

#ifdef ACC_T
typedef ACC_T acc_type;
#else
typedef float acc_type;
#endif


int main(int argc, char * argv[])
{
    int m = 16 * 256;
    int n = 16 * 256;
    int k = 16 * 256;

    int n_runs = 10;

    if (argc >= 2)
    {
        n_runs = atoi(argv[1]);
    }
    if (argc == 3)
    {
        int input_int = atoi(argv[2]);
        m = input_int;
        n = input_int;
        k = input_int;
    } else if (argc == 4)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else if (argc == 5)
    {
        n_runs = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }


    TimeMeasurement t;

    // Define matrices
    RandomMatrix<element_type, 2> A;
    RandomMatrix<element_type, 2> B;
    RandomMatrix<acc_type, 2> A_accT;
    RandomMatrix<acc_type, 2> B_accT;
    RandomMatrix<acc_type, 2> C;
    RandomMatrix<acc_type, 2> C_target;

    // Initialize matrices
    A.fill_rand<float_range>(m, k);
    B.fill_rand<float_range>(k, n);
    A_accT.fill_from(A, m, k);
    B_accT.fill_from(B, k, n);

    benchmark_kernel<acc_type, acc_type, 2, mm_kernel::register_tiled, false>(
        n_runs, m, n, k, A_accT, B_accT, C_target, C_target, std::string("GPU register tiled")
    );

    // TODO: make this work for Cutlass half, or cast?
//    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_naive, true>(
//        n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor naive")
//    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cublas, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("cublas")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_optimized, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor optimized")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cutlass_default, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("Cutlass default")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cutlass_mm, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("Cutlass")
    );

    cudaFree(A.to_gpu());
    cudaFree(B.to_gpu());
    cudaFree(C.to_gpu());
    cudaFree(C_target.to_gpu());
    cudaFree(A_accT.to_gpu());
    cudaFree(B_accT.to_gpu());

    return 0;
}
