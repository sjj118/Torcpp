cmake_minimum_required(VERSION 3.15)
project(torcpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(test test/TensorTest.cpp)

add_executable(torcpp main.cpp)