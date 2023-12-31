//
// Created by Stefan on 12/20/2023.
//

#pragma once

#include "vulkan_utils.hpp"

namespace gfx {

class Swapchain {
 public:
  Swapchain();
  ~Swapchain();

  void Init();
  void Shutdown();

 public:
  vk::SwapchainKHR NativeSwapchain;
  VkFormat ImageFormat;

  std::vector<VkImage> Images;
  std::vector<VkImageView> ImageViews;
};

}  // namespace gfx
