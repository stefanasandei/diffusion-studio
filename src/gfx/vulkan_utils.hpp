//
// Created by Stefan on 12/24/2023.
//

#pragma once

#include <vulkan/vulkan.hpp>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define VK_CHECK(x)                                                    \
  do {                                                                 \
    vk::Result err = x;                                                \
    if (err != vk::Result::eSuccess) {                                 \
      std::cerr << "Detected Vulkan error: " << vk::to_string(err)     \
                << " at line " << __LINE__ << ", in file " << __FILE__ \
                << '\n';                                               \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

