//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"

namespace gfx {

class AllocatedImage {
 public:
  AllocatedImage();
  ~AllocatedImage();

  void SetAllocator(const VmaAllocator& allocator, const vk::Device& device);

  void Create(glm::ivec2 size, vk::Format chosen_format,
              VkImageUsageFlags usage);
  void Destroy();

 public:
  VkImage Image;
  VkImageView View;

  VmaAllocation Allocation;

  vk::Extent3D Extent;
  vk::Format Format;

 private:
  vk::Device m_Device;
  VmaAllocator m_Allocator;
};

void CopyImageToImage(vk::CommandBuffer cmd, vk::Image source,
                      vk::Image destination, vk::Extent3D image_size);

}  // namespace gfx