cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mma_kernels LANGUAGES CXX CUDA)

set(SOURCES 
    src/cutlass_prototype/main.cu)

set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
set(CUDA_ARCHITECTURES 80)
set (USE_CUDA on)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(DEBUG "Cutlass dir: ${CUTLASS_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(CUDAToolkit REQUIRED cublas)
target_link_libraries(${PROJECT_NAME} CUDA::cublas)

include_directories("${CUTLASS_DIR}/include")
