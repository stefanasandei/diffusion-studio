cmake_minimum_required(VERSION 3.25)
project(lib)

add_library(lib INTERFACE)

# sd
set(SD_CUBLAS ON)
add_subdirectory(sd)
target_link_libraries(lib INTERFACE stable-diffusion)
target_include_directories(lib INTERFACE sd)

# stb
target_include_directories(lib INTERFACE stb)

# glfw
add_subdirectory(glfw)
target_link_libraries(lib INTERFACE glfw)
target_include_directories(lib INTERFACE glfw/include)

# vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(lib INTERFACE glfw Vulkan::Vulkan)
target_include_directories(lib INTERFACE ${Vulkan_INCLUDE_DIRS})

# vk-bootstrap
add_subdirectory(vk-bootstrap)
target_link_libraries(lib INTERFACE vk-bootstrap::vk-bootstrap)
target_include_directories(lib INTERFACE vk-bootstrap/src)

# vma
target_include_directories(lib INTERFACE vma/include)

# imgui
add_library(imgui STATIC imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_vulkan.cpp)
target_link_libraries(lib INTERFACE imgui)
target_include_directories(lib INTERFACE imgui)
target_include_directories(imgui PUBLIC imgui glfw/include ${Vulkan_INCLUDE_DIRS})
target_include_directories(lib INTERFACE imgui/backends)
