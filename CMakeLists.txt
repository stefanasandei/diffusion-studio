cmake_minimum_required(VERSION 3.25)
project(diffusion_studio)

set(CMAKE_CXX_STANDARD 23)

add_executable(diffusion_studio src/main.cpp)

add_subdirectory(lib)

target_link_libraries(diffusion_studio PRIVATE lib)

include_directories(src)
