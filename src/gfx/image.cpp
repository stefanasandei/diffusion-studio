//
// Created by Stefan on 12/29/2023.
//

#include "gfx/image.hpp"
#include "gfx/allocated_buffer.hpp"

#include "global.hpp"

namespace gfx {

AllocatedImage CreateImage(glm::ivec2 size, vk::Format format,
                           VkImageUsageFlags usage) {
  AllocatedImage newImage;
  newImage.SetAllocator(global.renderer->GetAllocator(),
                        global.context->Device);
  newImage.Create(size, format, usage);

  global.context->DeletionQueue.Push([=]() { DestroyImage(newImage); });

  return newImage;
}

AllocatedImage CreateImage(void* data, glm::ivec2 size, vk::Format format,
                           VkImageUsageFlags usage) {
  size_t data_size = size.x * size.y * 4;
  AllocatedBuffer uploadBuffer = CreateBuffer(
      global.renderer->GetAllocator(), data_size,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_TO_GPU);

  void* mappedData;
  vmaMapMemory(global.renderer->GetAllocator(), uploadBuffer.Allocation,
               &mappedData);
  memcpy(mappedData, data, data_size);
  vmaUnmapMemory(global.renderer->GetAllocator(), uploadBuffer.Allocation);

  AllocatedImage new_image =
      CreateImage(size, format,
                  usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  global.renderer->ImmediateSubmit([&](vk::CommandBuffer cmd) {
    TransitionImage(cmd, new_image.Image, vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eTransferDstOptimal);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = VkExtent3D{static_cast<uint32_t>(size.x),
                                        static_cast<uint32_t>(size.y), 1};

    vkCmdCopyBufferToImage(cmd, uploadBuffer.Buffer, new_image.Image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);

    TransitionImage(cmd, new_image.Image, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eReadOnlyOptimal);
  });

  vmaDestroyBuffer(global.renderer->GetAllocator(), uploadBuffer.Buffer,
                   uploadBuffer.Allocation);

  return new_image;
}

void UpdateImageData(void* data, AllocatedImage image) {
  size_t data_size = image.Extent.width * image.Extent.height * 4;
  AllocatedBuffer uploadBuffer = CreateBuffer(
      global.renderer->GetAllocator(), data_size,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_TO_GPU);

  void* mappedData;
  vmaMapMemory(global.renderer->GetAllocator(), uploadBuffer.Allocation,
               &mappedData);
  memcpy(mappedData, data, data_size);
  vmaUnmapMemory(global.renderer->GetAllocator(), uploadBuffer.Allocation);

  global.renderer->ImmediateSubmit([&](vk::CommandBuffer cmd) {
    TransitionImage(cmd, image.Image, vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eTransferDstOptimal);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = VkExtent3D{static_cast<uint32_t>(image.Extent.width),
                                        static_cast<uint32_t>(image.Extent.height), 1};

    vkCmdCopyBufferToImage(cmd, uploadBuffer.Buffer, image.Image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);

    TransitionImage(cmd, image.Image, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageLayout::eReadOnlyOptimal);
  });

  vmaDestroyBuffer(global.renderer->GetAllocator(), uploadBuffer.Buffer,
                   uploadBuffer.Allocation);
}

void DestroyImage(const AllocatedImage& img) {
  global.context->Device.destroyImageView(img.View);
  vmaDestroyImage(global.renderer->GetAllocator(), img.Image, img.Allocation);
}

}  // namespace gfx
