#include <iostream>
#include <cassert>
#include <cute/tensor.hpp>
#include "helpers.h"
#include "mma.cuh"
#include <cublas_v2.h>
#include <helpers.cuh>

#define SIZE 256

using namespace cute;

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
cublasStatus_t cublas_wrapper<half, half>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half *alpha,
        const half *A, int lda,
        const half *B, int ldb,
        const half *beta,
        half *C, int ldc
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
cublasStatus_t cublas_wrapper<half, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const half *A, int lda,
        const half *B, int ldb,
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
        std::cout << "CUBLAS initialization failed\n";
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
        std::cout << "CUBLAS error\n";
        printf("%s\n", cublasGetStatusName(stat));
        printf("%s\n", cublasGetStatusString(stat));
        exit(1);
    }

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

/*
Shape: Defines a coordinate space
Stride: Defines an index space.
Layout: Map between shapes by strides.
*/

template <class TA, class TB, class TC>
void run_mmm(
    int m, int n, int k,
    int ldA, int ldB, int ldC,
    TA *A, TB *B, TC *C)
{
	auto prob_shape = make_shape(m, n, k);

	// Make stride 1 in the reduction dimension.
	// That way we reduce over the K dimension.
	auto strideA = make_stride(ldA, Int<1>{}); // (M, K)
	auto strideB = make_stride(ldB, Int<1>{}); // (N, K)
	auto strideC = make_stride(Int<1>{}, ldC); // (M, N)

	// Make a threadblock of 128x128 over the output matrix.
	// The block then reduces 8 elements.
	auto blockM = Int<128>{};
	auto blockN = Int<128>{};
	auto blockK = Int<8>{};
	auto cta_tile = make_shape(blockM, blockN, blockK); // (BLK_M, BLK_N, BLK_K)

	// Make the layout for shared memory. We make it stride 1 in K and this k-major
	auto sA = make_layout(make_shape(blockM, blockK), LayoutRight{}); 
	auto sB = make_layout(make_shape(blockN, blockK), LayoutRight{});
	auto sC = make_shape(make_shape(blockM, blockN), LayoutRight{});

	// Define a layout of threads to do a copy to shared memory.
	// Each tile is 128x8, and so each thread needs to copy 4x1 elements.
	// This means we get vectorized loads.
	auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); //(M,K)
	auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); //(N,K)
	// We have the output BLK is (BLK_M, BLK_N)=(128, 126).
	// With the below layout of threads, we make each thread compute an 8x8 result
	auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // m-major

	cudaStream_t stream = 0;
	dim3 dimBlock(size(tC));
	dim3 dimGrid(size(ceil_div(m, blockM)),
				 size(ceil_div(n, blockN)));

	std::cout << "Launching kernel.\n"
			  << "Grid: " << dimGrid
			  << " Block: " << dimBlock
			  << std::endl;
	mmm_kernel<<<dimGrid, dimBlock, 0, stream>>>(
		prob_shape, cta_tile,
		A, strideA, sA, tA,
		B, strideB, sB, tB,
		C, strideC, sC, tC,
		1.0f, 0.0f);
	cudaDeviceSynchronize();
	gpuAssert(cudaPeekAtLastError());	
}


int main() 
{
    int M = SIZE;
    int N = SIZE;
    int K = SIZE;
    constexpr long n_runs = 10;
    unsigned long total_ops = (double)n_runs * 2.0 * M * N *K;
    
    RandomMatrix<float, 2> A; 
    RandomMatrix<float, 2> B;
    RandomMatrix<float, 2> C;

    RandomMatrix<float, 2> A_target;
    RandomMatrix<float, 2> B_target;
    RandomMatrix<float, 2> C_target;

    A.fill_rand<float_range>(M, K); A_target.fill_from(A, M, K);
    B.fill_rand<float_range>(K, N); B_target.fill_from(B, K, N);
    C.fill_zeros(M, N); C_target.fill_zeros(M, N);

    float *C_target_device = C_target.to_gpu();
    float *C_device = C.to_gpu(); 

	run_mmm(
		M, N, K,
		SIZE, SIZE, SIZE,
		A.to_gpu(), B.to_gpu(), C_device);
	cudaMemcpy(C.to_cpu(),
			   C_device,
			   C.flatSize() * sizeof(float),
			   cudaMemcpyDeviceToHost);
	
	benchmark_cublas(1,
					 A_target.to_gpu(), B_target.to_gpu(), C_target_device,
					 M, N, K);
	cudaMemcpy(C_target.to_cpu(),
			   C_target_device,
			   C_target.flatSize() * sizeof(float),
			   cudaMemcpyDeviceToHost);
	
	auto A_cpu = A.to_cpu();
	auto B_cpu = B.to_cpu();
	auto C_cpu = C.to_cpu();
	float *C_target_cpu = C_target.to_cpu();
	for (int i = 0; i < 3; i++)
	{
		std::cout << "A: " << A_cpu[i]
				  << " B: " << B_cpu[i]
				  << " C: " << C_cpu[i]
				  << " C_target: " << C_target_cpu[i]
				  << std::endl;
	}		    

	float *C_expected = C_target.to_cpu();
	float *C_actual = C.to_cpu();		

	Validator validator(C.to_cpu(), C_target.to_cpu(), M * N);
	validator.setEps(0.0005);
	validator.validate();
	
    // long time_us = benchmark_cublas(n_runs, A.to_gpu(), B.to_gpu(), C.to_gpu(), M, N, K);
    // printGFlops(time_us, total_ops);

    return 0;
}
