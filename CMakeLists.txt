cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mma_kernels LANGUAGES CXX CUDA)

set(SOURCES 
    src/cutlass_prototype/main.cu)
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
set(CUDA_PROTOTYPE_DIR "${CMAKE_SOURCE_DIR}/src/cuda_prototype")
set(CUDA_ARCHITECTURES 80)
set (USE_CUDA on)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -arch=sm_80 -Xptxas=-v")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message("Cutlass dir: ${CUTLASS_DIR}")
message("${CUDA_PROTOTYPE_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(CUDAToolkit REQUIRED cublas)
target_link_libraries(${PROJECT_NAME} CUDA::cublas)

include_directories(
    "${CUTLASS_DIR}/include" 
    "${CUDA_PROTOTYPE_DIR}")
