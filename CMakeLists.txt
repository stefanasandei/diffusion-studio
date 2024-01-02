cmake_minimum_required(VERSION 3.25)
project(diffusion_studio)

set(CMAKE_CXX_STANDARD 23)

add_executable(diffusion_studio src/main.cpp
        src/platform/window.h
        src/util/std.hpp
        src/util/error.cpp
        src/platform/window.cpp
        src/gfx/gfx.hpp
        src/gfx/renderer.hpp
        src/gfx/vulkan_utils.hpp
        src/gfx/context.hpp
        src/global.hpp
        src/gfx/renderer.cpp
        src/global.cpp
        src/gfx/context.cpp
        src/memory/deletion_queue.hpp
        src/memory/deletion_queue.cpp
        src/gfx/swapchain.hpp
        src/gfx/swapchain.cpp
        src/gfx/allocated_image.hpp
        src/gfx/allocated_image.cpp
        src/gfx/vma_impl.cpp
        src/gfx/imgui_layer.hpp
        src/gfx/imgui_layer.cpp
        src/gfx/shader.hpp
        src/gfx/shader.cpp
        src/gfx/allocated_buffer.hpp
        src/gfx/allocated_buffer.cpp
        src/gfx/descriptors.hpp
        src/gfx/descriptors.cpp
        src/gfx/image.hpp
        src/gfx/image.cpp
        src/gfx/stb_impl.cpp
        src/app/application.hpp
        src/app/application.cpp
        src/dl/diffusionmodel.hpp
        src/app/panels/panel.hpp
        src/dl/sdinstance.hpp
        src/app/panels/create_panel.hpp
        src/app/panels/create_panel.hpp
        src/app/panels/create_panel.cpp
        src/dl/diffusionmodel.cpp
        src/app/panels/preview_panel.cpp
        src/dl/sdinstance.cpp
        src/util/safe_queue.hpp
        src/gfx/imgui_stdlib.hpp
        src/gfx/imgui_stdlib.cpp)

add_subdirectory(lib)

target_link_libraries(diffusion_studio PRIVATE lib)

include_directories(src)
