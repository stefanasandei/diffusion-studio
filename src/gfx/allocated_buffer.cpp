//
// Created by Stefan on 12/26/2023.
//

#include "gfx/allocated_buffer.hpp"

namespace gfx {

AllocatedBuffer CreateBuffer(VmaAllocator allocator, size_t alloc_size,
                             vk::BufferUsageFlags usage,
                             VmaMemoryUsage memory_usage) {
  AllocatedBuffer res;

  vk::BufferCreateInfo buffer_info;
  buffer_info.setSize(alloc_size);
  buffer_info.setUsage(usage);

  VmaAllocationCreateInfo vma_alloc_info = {};
  vma_alloc_info.usage = memory_usage;
  vma_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  vmaCreateBuffer(allocator,
                  reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info),
                  &vma_alloc_info, reinterpret_cast<VkBuffer*>(&res.Buffer),
                  &res.Allocation, &res.Info);

  return res;
}

}  // namespace gfx
