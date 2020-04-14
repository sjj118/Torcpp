cmake_minimum_required(VERSION 3.15)
project(ctorch)

set(CMAKE_CXX_STANDARD 20)

add_executable(ctorch main.cpp Tensor.cpp Tensor.h)

add_compile_options(-fno-elide-constructors)