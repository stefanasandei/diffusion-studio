//
// Created by Stefan on 12/26/2023.
//

#include "gfx/allocated_image.hpp"

namespace gfx {

AllocatedImage::AllocatedImage() {}

AllocatedImage::~AllocatedImage() {}

void AllocatedImage::Create(glm::ivec2 size, vk::Format chosen_format,
                            VkImageUsageFlags usage) {
  Extent = vk::Extent3D{static_cast<uint32_t>(size.x),
                        static_cast<uint32_t>(size.y), 1};

  Format = chosen_format;

  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = nullptr;

  info.imageType = VK_IMAGE_TYPE_2D;

  info.format = static_cast<VkFormat>(Format);
  info.extent = Extent;

  info.mipLevels = 1;
  info.arrayLayers = 1;

  info.samples = VK_SAMPLE_COUNT_1_BIT;

  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;

  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  alloc_info.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vmaCreateImage(m_Allocator, &info, &alloc_info, &Image, &Allocation, nullptr);

  VkImageViewCreateInfo view_info = {};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;

  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.image = Image;
  view_info.format = static_cast<VkFormat>(Format);
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;
  view_info.subresourceRange.aspectMask =
      (chosen_format == vk::Format::eD32Sfloat
           ? VK_IMAGE_ASPECT_DEPTH_BIT
           : VK_IMAGE_ASPECT_COLOR_BIT);

  VkResult res = vkCreateImageView(m_Device, &view_info, nullptr, &View);
  if (res != VK_SUCCESS) {
    printf("Failed to create a VkImageView for an allocated image.\n");
    std::exit(1);
  }
}

void AllocatedImage::Destroy() {
  vkDestroyImageView(m_Device, View, nullptr);
  vmaDestroyImage(m_Allocator, Image, Allocation);
}

void AllocatedImage::SetAllocator(VmaAllocator const& allocator,
                                  const vk::Device& device) {
  m_Allocator = allocator;
  m_Device = device;
}

void CopyImageToImage(vk::CommandBuffer cmd, vk::Image source,
                      vk::Image destination, vk::Extent3D image_size) {
  VkImageBlit2 blitRegion{.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
                          .pNext = nullptr};

  blitRegion.srcOffsets[1].x = image_size.width;
  blitRegion.srcOffsets[1].y = image_size.height;
  blitRegion.srcOffsets[1].z = 1;

  blitRegion.dstOffsets[1].x = image_size.width;
  blitRegion.dstOffsets[1].y = image_size.height;
  blitRegion.dstOffsets[1].z = 1;

  blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blitRegion.srcSubresource.baseArrayLayer = 0;
  blitRegion.srcSubresource.layerCount = 1;
  blitRegion.srcSubresource.mipLevel = 0;

  blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blitRegion.dstSubresource.baseArrayLayer = 0;
  blitRegion.dstSubresource.layerCount = 1;
  blitRegion.dstSubresource.mipLevel = 0;

  VkBlitImageInfo2 blitInfo{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                            .pNext = nullptr};

  blitInfo.dstImage = destination;
  blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  blitInfo.srcImage = source;
  blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  blitInfo.filter = VK_FILTER_LINEAR;
  blitInfo.regionCount = 1;
  blitInfo.pRegions = &blitRegion;

  vkCmdBlitImage2(cmd, &blitInfo);
}

}  // namespace gfx
