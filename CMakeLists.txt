cmake_minimum_required(VERSION 3.12)

project(cuda LANGUAGES C CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(USE_CUDA ON)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)

add_subdirectory(tasks)

add_executable(cuda cuda.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda PRIVATE "-G")
endif()

# ------------------------------------

# add_executable(app main.cu)

# add_executable(app_1 aboba.cpp)

# foreach(entry IN LISTS TASK_INCLUDE_DIRS) 
#     message(STATUS ${entry})
# endforeach()
