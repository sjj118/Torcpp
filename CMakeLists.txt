cmake_minimum_required(VERSION 3.15)
project(torcpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(test test/TensorTest.cpp core/Tensor.cpp core/random.cpp)

add_executable(torcpp main.cpp core/Tensor.cpp core/random.cpp)