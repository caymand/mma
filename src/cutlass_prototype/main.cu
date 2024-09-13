#include <iostream>
#include <cassert>
#include <cute/tensor.hpp>
#include "helpers.h"
#include "mma.cuh"
#include <cublas_v2.h>


// using namespace cute;

#define SIZE 4096

using namespace cute;

template <class TElm, class TAcc>
cublasStatus_t cublas_wrapper(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const TElm *alpha,
        const TElm *A, int lda,
        const TElm *B, int ldb,
        const TElm *beta,
        TAcc *C, int ldc
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

/*
Shape: Defines a coordinate space
Stride: Defines an index space.
Layout: Map between shapes by strides.
*/


template <class TA, class TB, class TC>
void run_mma(
    int m, int n, int k,
    int ldA, int ldB, int ldC,
    TA *A, TB *B, TC *C)
{
    auto prob_shape = make_shape(m, n, k);    // (M, N, K)

    // cute::Stride. This defines TT stides (row major A and row major B)
    // CuTe expect the matrices to be stored: (M,K) x (N, K).
    // The reduction is over the K dimension, so this is where we stride by 1.
    // And each row is ldA wide.
    // ldA to get to next row. Stide 1 to get to next column elm.
    auto strideA = make_stride(ldA, Int<1>{}); // (M, K)
    auto strideB = make_stride(Int<1>{}, ldA); // (K, N)
    auto strideC = make_stride(ldC, Int<1>{}); // (M, N)

    // Tile sizes for the 1block. CuTe calls a threadblock CTA (Cuda Thread Array)
    auto tileM = Int<128>{};
    auto tileN = Int<128>{};
    auto tileK = Int<8>{};

    // Define tall thin tiles in A and long short tiles in B    
    auto cta_tiler_shape = make_shape(tileM, tileN, tileK);

    auto sharedA = make_layout(make_shape(tileM, tileK), LayoutRight{}); // (m,k)
    auto sharedB = make_layout(make_shape(tileK, tileN), LayoutRight{}); // (k, n)
    auto sharedC = make_layout(make_shape(tileM, tileN), LayoutRight{});

    // Give each thread a small 4x1 chunk to copy. TODO: does this mean all 8 elements in k dimens and 4 rows?
    // This is the organization of threads we want to use to load the chunk of memory used
    // by the CTA into shared memory.
    auto loadTileA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
    auto loadTileB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
    // This means that each thread needs to compute a 8x8 tile of C
    auto threadLayoutC = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{});

    // gemm_device(
    //     prob_shape, cta_tiler_shape,
    //     A, strideA, sharedA, loadTileA,
    //     B, strideB, sharedB, loadTileB,
    //     C, strideC, sharedC, threadLayoutC        
    // );
    // A CuTe layout is a tuple of (Shape, Stride).
}

int main() 
{
    int M = SIZE;
    int N = SIZE;
    int K = SIZE;
    half *h_A, *h_B, *h_C;
    half *d_A, *d_B, *d_C;

    h_A = (half *)malloc(M * K); assert(h_A != nullptr);
    h_B = (half *)malloc(M * K); assert(h_B != nullptr);
    h_C = (half *)malloc(M * K); assert(h_C != nullptr);

    create_matrix(M, K, h_A); copy_device(d_A, h_A, M*K);
    create_matrix(K, N, h_B); copy_device(d_B, h_B, K*N);
    assert(memset(h_C, half(), sizeof(half) * M * N) != nullptr);
    copy_device(d_C, h_C, M*N);

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    half alpha = (half) 1.0;
    half beta = (half) 0.0;
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    stat = cublas_wrapper(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha,
        // Cublas uses column major, so we need to swap A and B, since B^T @ A^T = (A @ B)^T = C^T
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS error\n");
        printf("%s\n", cublasGetStatusName(stat));
        printf("%s\n", cublasGetStatusString(stat));
        exit(1);
    }


    return 0;
}