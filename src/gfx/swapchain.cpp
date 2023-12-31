//
// Created by Stefan on 12/26/2023.
//

#include "gfx/swapchain.hpp"

#include "global.hpp"

namespace gfx {

Swapchain::Swapchain() {
  Init();

  global.context->DeletionQueue.Push([&]() { Shutdown(); });
}

Swapchain::~Swapchain() {}

void Swapchain::Init() {
  vkb::SwapchainBuilder swapchain_builder(global.context->PhysicalDevice,
                                          global.context->Device,
                                          global.context->Surface);

  vkb::Swapchain vkbSwapchain =
      swapchain_builder.use_default_format_selection()
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(global.window->GetSize().x,
                              global.window->GetSize().y)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build()
          .value();

  NativeSwapchain = vkbSwapchain.swapchain;
  Images = vkbSwapchain.get_images().value();
  ImageViews = vkbSwapchain.get_image_views().value();

  ImageFormat = vkbSwapchain.image_format;
}

void Swapchain::Shutdown() {
  for (auto& image_view : ImageViews) {
    global.context->Device.destroyImageView(image_view);
  }

  global.context->Device.destroySwapchainKHR(NativeSwapchain);
}

}  // namespace gfx
