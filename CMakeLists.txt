cmake_minimum_required(VERSION 3.12)

project(cuda LANGUAGES C CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(USE_CUDA ON)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)

# -------------------------------------

add_subdirectory(deps)
add_subdirectory(rng)

add_subdirectory(tasks)

add_subdirectory(tests)
