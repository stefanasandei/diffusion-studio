//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"

namespace gfx {

struct AllocatedBuffer {
  vk::Buffer Buffer;
  VmaAllocation Allocation;
  VmaAllocationInfo Info;
};

[[nodiscard]] AllocatedBuffer CreateBuffer(VmaAllocator allocator, size_t alloc_size, vk::BufferUsageFlags usage,
                             VmaMemoryUsage memory_usage);

}  // namespace gfx
