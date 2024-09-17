#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <chrono>

#include <cuda_runtime.h>
#include <cstdarg>
#include "cutlass/detail/helper_macros.hpp"


// TODO: check this
CUTLASS_HOST_DEVICE
float operator+(half const& lhs, float const& rhs) {
    return __half2float(lhs) + rhs;
}


CUTLASS_HOST_DEVICE
float operator+(float const& lhs, half const& rhs) {
    return lhs + __half2float(rhs);
}


constexpr int float_range = RAND_MAX ;

int gpuAssert(cudaError_t code);

template <typename T, int N>
class RandomMatrix
{
    private:
        unsigned seed;
        void * gpu_mem = nullptr;
        std::vector<T> flatMat;
        std::vector<unsigned> dimensions;        
        void setDimensions(unsigned first_dim, va_list dimensions);

    public:                
        RandomMatrix();        
        T* to_cpu();
        T* to_gpu();
        unsigned flatSize();
        RandomMatrix<T, N>& setSeed(unsigned s);        
        template <int R> void fill_rand(const unsigned dimensions, ...);
        template <typename U> void fill_from(RandomMatrix<U, N> &other, const unsigned dimensions, ...);
        void fill_zeros(const unsigned int dimensions, ...);
};


template <typename T>
class Validator
{
    private:
        std::vector<T> flatMat1;
        std::vector<T> flatMat2;
        T eps;
    public:
        Validator(T *flatMat1, T *flatMat2, int N);
        void setEps(T);
        void validate();
};


class TimeMeasurement
{
    private:
        int resolution;
        std::chrono::steady_clock::time_point tstart;
        std::chrono::steady_clock::time_point tend;  
    public:
        TimeMeasurement(int resolution = 1000000);
        int start();
        int stop();
        long int elapsed();
        
};

int printPerformanceMetric(long int elapsed_time_us, unsigned long total, const char *metric_name) 
{
//    double elapsed_sec = elapsed_time_us * 1e-6f;
//    double metric = (total / elapsed_sec) * 1.0e-9f; // make it in Giga scale

//    double elapsed_sec = elapsed_time_us * 1e-6f;
    double metric = (total / elapsed_time_us) * 1.0e-3f; // make it in Giga scale
    std::cout << metric_name << " : " << metric << std::endl;
    return metric < 0;
}
int printGBSec(long int elapsed_time_us, unsigned long total_memsize) 
{
    return printPerformanceMetric(elapsed_time_us, total_memsize, "GB/sec");
}

int printGFlops(long int elapsed_time_us, unsigned long total_flops) 
{
    return printPerformanceMetric(elapsed_time_us, total_flops, "GFlops/sec");
}

int gpuAssert(cudaError_t code) 
{
    if(code != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(code));
        exit(1);
    }
    return 1;
}


// Validator
template <typename T>
Validator<T>::Validator(T *flatMat1, T* flatMat2, int N)
{
    this->flatMat1.insert(this->flatMat1.begin(), flatMat1, flatMat1 + N);
    this->flatMat2.insert(this->flatMat2.begin(), flatMat2, flatMat2 + N);
    this->eps = 0.0000000001; // e-10
}
template <typename T>
void Validator<T>::setEps(T eps) {
    this->eps = eps;
}

template <typename T>
void Validator<T>::validate()
{
    bool errValue = false;
    std::vector<bool> flatMatrixMask(this->flatMat1.capacity());
    std::transform(
            this->flatMat1.begin(), this->flatMat1.end(), this->flatMat2.begin(), flatMatrixMask.begin(),
            [this] (T v1, T v2) {
//            float err = (float)std::abs(v1 - v2) / std::max(v1, v2);
//            return err < this->eps;

//            Calculate err without using abs
                T err = (v1 - v2) / std::max(v1, v2);

//            printf("v1: %.5f, v2: %.5f, err: %.5f\n", (float) v1, (float) v2, (float) err);

                return err < (T) 0.0 ? -err < this->eps : err < this->eps;
            }
    );
    auto errorCount = std::count(flatMatrixMask.begin(), flatMatrixMask.end(), errValue);
    auto firstError = std::find(flatMatrixMask.begin(), flatMatrixMask.end(), errValue) - flatMatrixMask.begin();
    if (errorCount <= 0) {
        std::cout << "VALID" << std::endl;
        std::cout << "-----" << std::endl;
        return;
    }
    std::cout << "INVALID" << std::endl;
    std::cout << "-------" << std::endl;
    std::printf("Found: %d wrong elements\n", errorCount);
    std::cout << "First at flat index: " << firstError << std::endl;
    std::printf("Expected %.5f, got %.5f\n", (float) this->flatMat1[firstError], (float) this->flatMat2[firstError]);
}

// RandomMatrix

//Private
template <typename T, int N>
unsigned RandomMatrix<T, N>::flatSize()
{
    unsigned acc = 1;

    for (int dim : this->dimensions) {
        acc *= dim;
    }

    return acc;
}
template <typename T, int N>
void RandomMatrix<T, N>::setDimensions(unsigned first_dim, va_list dimensions)
{
    this->dimensions.clear();
    this->dimensions.push_back(first_dim);
    for (int i = 0; i < N - 1; i++)
    {
        unsigned dim = va_arg(dimensions, unsigned);
        this->dimensions.push_back(dim);
    }
}

template <typename T, int N>
RandomMatrix<T, N>::RandomMatrix()
{
    this->setSeed(37);
}

template <typename T, int N>
template <typename U>
void RandomMatrix<T, N>::fill_from(RandomMatrix<U, N> &other, const unsigned first_dim, ...)
{
    va_list remaining_dims; va_start(remaining_dims, first_dim);
    this->setDimensions(first_dim, remaining_dims);
    U *other_flat_mat = other.to_cpu();
    int size = other.flatSize();
    for (int i = 0; i < size; i++) {
        U v = other_flat_mat[i];
        this->flatMat.push_back((T) v);
    }
}

template <typename T, int N>
T* RandomMatrix<T, N>::to_cpu()
{
    return this->flatMat.data();
}

/*Copy host memory to gpu.*/
template <typename T, int N>
T* RandomMatrix<T, N>::to_gpu()
{
    size_t n_bytes = this->flatSize() * sizeof(T);

    if (this->gpu_mem == nullptr) {
        if (!gpuAssert(cudaMalloc(&gpu_mem, n_bytes)))
        {
            std::cout << "GPU memory allocation error" << std::endl;
        }
    }

    gpuAssert(cudaMemcpy(gpu_mem, this->to_cpu(), n_bytes, cudaMemcpyHostToDevice));
    return (T *) gpu_mem;
}

template <typename T, int N>
RandomMatrix<T, N>& RandomMatrix<T, N>::setSeed(unsigned s)
{
    srand(s);
    return *this;
}
template <typename T, int N>
template <int R>
void RandomMatrix<T, N>::fill_rand(const unsigned first_dim, ...)
{
    va_list remaining_dims;
    va_start(remaining_dims, first_dim);
    this->setDimensions(first_dim, remaining_dims);
    this->flatMat.resize(this->flatSize());
    std::cout << "Capacity: " << this->flatSize()  << std::endl;
    std::generate(this->flatMat.begin(), this->flatMat.end(), [](){
        return (T) (rand() / ((float) R));
    });
}

template <typename T, int N>
void RandomMatrix<T, N>::fill_zeros(const unsigned first_dim, ...)
{
    va_list remaining_dims;
    va_start(remaining_dims, first_dim);
    this->setDimensions(first_dim, remaining_dims);
    std::cout << "Capacity: " << this->flatSize()  << std::endl;
    this->flatMat.resize(this->flatSize());
    std::fill(this->flatMat.begin(), this->flatMat.end(), (T) 0);
}

// TimeMeasurment
TimeMeasurement::TimeMeasurement(int resolution) {
    this ->resolution = resolution;
}
int TimeMeasurement::start()
{
    tstart = std::chrono::steady_clock::now(); 
    return 0;
}
int TimeMeasurement::stop()
{
    tend = std::chrono::steady_clock::now(); 
    return 0;
}
long int TimeMeasurement::elapsed()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();    
}
