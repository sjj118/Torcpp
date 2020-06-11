cmake_minimum_required(VERSION 3.15)
project(torcpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(xor examples/xor.cpp core/Tensor.cpp core/random.cpp)

add_executable(tutorial examples/tutorial.cpp core/Tensor.cpp core/random.cpp)