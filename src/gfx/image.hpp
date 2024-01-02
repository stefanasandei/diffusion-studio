//
// Created by Stefan on 12/29/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"
#include "gfx/allocated_image.hpp"

namespace gfx {

AllocatedImage CreateImage(glm::ivec2 size, vk::Format format,
                           VkImageUsageFlags usage);
AllocatedImage CreateImage(void* data, glm::ivec2 size, vk::Format format,
                           VkImageUsageFlags usage);
void UpdateImageData(void* data, AllocatedImage image);
void DestroyImage(const AllocatedImage& img);

}  // namespace gfx
