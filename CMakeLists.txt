cmake_minimum_required(VERSION 3.20)
project(spmm LANGUAGES CXX CUDA)

add_executable(spmm spmm.cu)
target_compile_options(spmm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
