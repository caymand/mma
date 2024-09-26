cmake_minimum_required(VERSION 3.29)
project(cuda_prototype CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(cuda_prototype main.cu)

set_target_properties(cuda_prototype PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuda_prototype PRIVATE ../../cutlass/include/)
target_link_libraries(cuda_prototype -lcublas)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr> -lineinfo")
