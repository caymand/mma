#include <stdlib.h>     /* malloc, free, rand */


#ifndef RANDMAX
#define RANDMAX 10.0
#endif

template<typename T>
void create_matrix(int height, int width, T *matrix, int seed = 42)
{
    srand(seed);
    for (size_t i = 0; i < width * height; i++)
    {
        matrix[i] = static_cast<T>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));   
    }
}

int gpuAssert(cudaError_t code) 
{
    if(code != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(code));
        exit(1);
    }
    return 1;
}


template<typename T>
void copy_device(T* d_matrix, T *h_matrix , size_t n_bytes)
{
    if (h_matrix == nullptr) {
        if (!gpuAssert(cudaMalloc(&d_matrix, n_bytes)))
        {
            std::cout << "GPU memory allocation error" << std::endl;
        }
    }

    gpuAssert(cudaMemcpy(d_matrix, h_matrix, n_bytes, cudaMemcpyHostToDevice));
}

template <class T>
void validate(T *expected, T *actual, size_t flatSize)
{
    for (size_t i = 0; i < flatSize; i++)
    {
        if (expected[i] != actual[i])
        {
            printf("Wrong element at flat index %d\nExpected: ", i);
            std::cout << expected[i];
            std::cout << " got: ";
            std::cout << actual[i] << std::endl;
        }
    }
}