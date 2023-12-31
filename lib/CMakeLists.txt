cmake_minimum_required(VERSION 3.25)
project(lib)

add_library(lib INTERFACE)

# sd
set(SD_CUBLAS ON)
add_subdirectory(sd)
target_link_libraries(lib INTERFACE stable-diffusion)
target_include_directories(lib INTERFACE sd)
