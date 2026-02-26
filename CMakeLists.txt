cmake_minimum_required(VERSION 3.20)
project(spmm LANGUAGES CXX CUDA)

add_executable(spmm
    src/spmm.cu
    src/sparse_csr.cpp
)
target_include_directories(spmm PRIVATE include)
target_include_directories(spmm SYSTEM PRIVATE vendor)

target_compile_options(spmm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror -diag-suppress=20015,20012>
)

# add_subdirectory(external/eigen-5.0.0)
# target_link_libraries(spmm PRIVATE Eigen3::Eigen)
