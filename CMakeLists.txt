cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mma_kernels LANGUAGES CXX CUDA)

set(SOURCES 
    src/main.cu)

set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
set(CUDA_ARCHITECTURES 80)
set (USE_CUDA on)

message(DEBUG "Cutlass dir: ${CUTLASS_DIR}")

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories("${CUTLASS_DIR}/include")
