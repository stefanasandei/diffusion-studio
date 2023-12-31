//
// Created by Stefan on 12/24/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"
#include "memory/deletion_queue.hpp"

namespace gfx {

class Context {
 public:
  Context();
  ~Context();

  void Shutdown();
  void Init();

 public:
  vk::Instance Instance;
  vk::SurfaceKHR Surface;
  vk::PhysicalDevice PhysicalDevice;
  vk::Device Device;

  vk::Queue GraphicsQueue;
  uint32_t GraphicsQueueFamily;

  memory::DeletionQueue DeletionQueue;
};

VkSurfaceKHR CreateSurfaceGlfw(VkInstance instance, GLFWwindow* window);

vk::ImageSubresourceRange ImageSubresourceRange(
    vk::ImageAspectFlags aspect_mask);

void TransitionImage(vk::CommandBuffer cmd, vk::Image image,
                     vk::ImageLayout current_layout,
                     vk::ImageLayout new_layout);

}  // namespace gfx
