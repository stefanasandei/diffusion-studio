//
// Created by Stefan on 12/24/2023.
//

#include "gfx/context.hpp"

#include "global.hpp"

namespace gfx {

Context::Context() { Init(); }

Context::~Context() { Shutdown(); }

void Context::Shutdown() { DeletionQueue.Flush(); }

void Context::Init() {
  // create the instance
  vkb::InstanceBuilder builder;
  auto inst_ret = builder.set_app_name("Mesh Viewer")
                      .set_debug_messenger_severity(
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                      .set_debug_messenger_type(
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
                      .use_default_debug_messenger()
                      .require_api_version(1, 3, 0)
                      .build();
  util::error::ErrNDie(!inst_ret, "Failed to create a Vulkan instance");

  vkb::Instance vkb_inst = inst_ret.value();
  Instance = vkb_inst.instance;

  // create the surface
  Surface = CreateSurfaceGlfw(vkb_inst.instance, global.window->GetNative());
  DeletionQueue.Push([&]() { Instance.destroy(Surface); });

  // create the physical device
  vk::PhysicalDeviceVulkan13Features features13{};
  features13.setDynamicRendering(true);
  features13.setSynchronization2(true);

  vk::PhysicalDeviceVulkan12Features features12{};
  features12.setBufferDeviceAddress(true);
  features12.setDescriptorIndexing(true);

  vkb::PhysicalDeviceSelector selector{vkb_inst};
  auto phys_ret = selector.set_surface(Surface)
                      .set_minimum_version(1, 3)
                      .set_required_features_13(features13)
                      .set_required_features_12(features12)
                      .select();
  util::error::ErrNDie(!phys_ret, "Failed to create a Vulkan physical device");

  vkb::PhysicalDevice physical_device = phys_ret.value();
  PhysicalDevice = physical_device.physical_device;

  // create the device
  vkb::DeviceBuilder device_builder{physical_device};
  auto device_ret = device_builder.build();
  util::error::ErrNDie(!device_ret, "Failed to create a Vulkan device");

  Device = device_ret.value().device;

  GraphicsQueue =
      device_ret.value().get_queue(vkb::QueueType::graphics).value();
  GraphicsQueueFamily =
      device_ret.value().get_queue_index(vkb::QueueType::graphics).value();
}

VkSurfaceKHR CreateSurfaceGlfw(VkInstance instance, GLFWwindow* window) {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
  util::error::ErrNDie(err != VK_SUCCESS,
                       "Failed to create a Vulkan surface from GLFW.");
  return surface;
}

vk::ImageSubresourceRange ImageSubresourceRange(
    vk::ImageAspectFlags aspect_mask) {
  vk::ImageSubresourceRange sub_image;

  sub_image.setAspectMask(aspect_mask);
  sub_image.setBaseMipLevel(0);
  sub_image.setLevelCount(1);
  sub_image.setBaseArrayLayer(0);
  sub_image.setLayerCount(1);

  return sub_image;
}

void TransitionImage(vk::CommandBuffer cmd, vk::Image image,
                     vk::ImageLayout current_layout,
                     vk::ImageLayout new_layout) {
  vk::ImageMemoryBarrier2 image_barrier;

  image_barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands);
  image_barrier.setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite);
  image_barrier.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
  image_barrier.setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite |
                                 vk::AccessFlagBits2::eMemoryRead);

  image_barrier.setOldLayout(current_layout);
  image_barrier.setNewLayout(new_layout);

  vk::ImageAspectFlags aspect_mask =
      (new_layout == vk::ImageLayout::eDepthAttachmentOptimal)
          ? vk::ImageAspectFlagBits::eDepth
          : vk::ImageAspectFlagBits::eColor;

  image_barrier.setSubresourceRange(ImageSubresourceRange(aspect_mask));
  image_barrier.setImage(image);

  vk::DependencyInfo dep_info;
  dep_info.setImageMemoryBarrierCount(1);
  dep_info.setPImageMemoryBarriers(&image_barrier);

  cmd.pipelineBarrier2(&dep_info);
}

vk::ImageSubresourceRange image_subresource_range(
    vk::ImageAspectFlags aspect_mask) {
  vk::ImageSubresourceRange sub_image;

  sub_image.setAspectMask(aspect_mask);
  sub_image.setBaseMipLevel(0);
  sub_image.setLevelCount(1);
  sub_image.setBaseArrayLayer(0);
  sub_image.setLayerCount(1);

  return sub_image;
}

}  // namespace gfx
