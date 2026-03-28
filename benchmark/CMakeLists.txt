cmake_minimum_required(VERSION 3.21)
project(sptc_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)

# Dense baseline benchmark
add_executable(benchmark_sptc benchmark_main.cpp)
target_link_libraries(benchmark_sptc PRIVATE ${BLAS_LIBRARIES})
target_include_directories(benchmark_sptc PRIVATE "${CMAKE_SOURCE_DIR}")

# TiledArray distributed benchmark
find_package(tiledarray QUIET
    HINTS "${CMAKE_SOURCE_DIR}/../tiledarray/install/lib/cmake/tiledarray")
if(tiledarray_FOUND)
    message(STATUS "TiledArray found — building ta_benchmark")
    add_executable(ta_benchmark ta_benchmark_main.cpp)
    target_link_libraries(ta_benchmark PRIVATE tiledarray)
    target_include_directories(ta_benchmark PRIVATE "${CMAKE_SOURCE_DIR}")

    add_executable(ta_smoke_test ta_smoke_test.cpp)
    target_link_libraries(ta_smoke_test PRIVATE tiledarray)
else()
    message(STATUS "TiledArray not found — skipping ta_benchmark")
endif()
