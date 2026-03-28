cmake_minimum_required(VERSION 3.21)
project(sptc_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)

add_executable(benchmark_sptc benchmark_main.cpp)
target_link_libraries(benchmark_sptc PRIVATE ${BLAS_LIBRARIES})
target_include_directories(benchmark_sptc PRIVATE "${CMAKE_SOURCE_DIR}")
