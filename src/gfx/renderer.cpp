//
// Created by Stefan on 12/24/2023.
//

#include "gfx/renderer.hpp"

#include "global.hpp"

#include "gfx/shader.hpp"
#include "gfx/image.hpp"

#include <stb_image.h>
#include <imgui_impl_vulkan.h>
#include <glm/glm.hpp>

namespace gfx {

Renderer::Renderer() {
  m_FramesCount = 0;

  InitCommands();
  InitSyncStructures();
  InitVma();
  InitDrawTarget();
  InitDescriptors();
  InitPipelines();

  m_StartTime = std::chrono::high_resolution_clock::now();
}

Renderer::~Renderer() {
  global.context->Device.waitIdle();

  m_DrawImage.Destroy();

  global.context->DeletionQueue.Flush();

  for (std::size_t i = 0; i < FRAME_COUNT; i++) {
    global.context->Device.destroyCommandPool(m_Frames[i].CmdPool);

    global.context->Device.destroyFence(m_Frames[i].RenderFence);
    global.context->Device.destroySemaphore(m_Frames[i].RenderSemaphore);
    global.context->Device.destroySemaphore(m_Frames[i].SwapchainSemaphore);
  }
}

void Renderer::Draw() {
  // wait & reset the fence
  VK_CHECK(global.context->Device.waitForFences(1, &GetFrame().RenderFence,
                                                true, 1000000000));

  GetFrame().DeletionQueue.Flush();
  GetFrame().Descriptors.ClearPools();

  VK_CHECK(global.context->Device.resetFences(1, &GetFrame().RenderFence));

  // grab the swapchain image
  uint32_t swapchain_image_index = 0;
  vk::SwapchainKHR native_swapchain =
      *static_cast<vk::SwapchainKHR*>(&global.swapchain->NativeSwapchain);
  vk::Result res = global.context->Device.acquireNextImageKHR(
      native_swapchain, 1000000000, GetFrame().SwapchainSemaphore, nullptr,
      &swapchain_image_index);
  if (res == vk::Result::eErrorOutOfDateKHR) {
    Resize();
    return;
  }

  vk::Image current_img = global.swapchain->Images[swapchain_image_index];

  vk::ImageView current_img_view =
      global.swapchain->ImageViews[swapchain_image_index];

  // reset the main cmd buffer
  GetFrame().MainCmdBuffer.reset();

  vk::CommandBufferBeginInfo cmd_begin_info;
  cmd_begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  VK_CHECK(GetFrame().MainCmdBuffer.begin(&cmd_begin_info));

  // start drawing
  TransitionImage(GetFrame().MainCmdBuffer, m_DrawImage.Image,
                  vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

  DrawBackground(GetFrame().MainCmdBuffer, m_DrawImage.Image);

  TransitionImage(GetFrame().MainCmdBuffer, m_DrawImage.Image,
                  vk::ImageLayout::eGeneral,
                  vk::ImageLayout::eTransferSrcOptimal);

  TransitionImage(GetFrame().MainCmdBuffer, current_img,
                  vk::ImageLayout::eUndefined,
                  vk::ImageLayout::eTransferDstOptimal);

  CopyImageToImage(GetFrame().MainCmdBuffer, m_DrawImage.Image, current_img,
                   m_DrawImage.Extent);

  TransitionImage(GetFrame().MainCmdBuffer, current_img,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageLayout::eColorAttachmentOptimal);

  DrawImGui(GetFrame().MainCmdBuffer, current_img_view);

  TransitionImage(GetFrame().MainCmdBuffer, m_DrawImage.Image,
                  vk::ImageLayout::eTransferSrcOptimal,
                  vk::ImageLayout::eTransferDstOptimal);

  TransitionImage(GetFrame().MainCmdBuffer, current_img,
                  vk::ImageLayout::eColorAttachmentOptimal,
                  vk::ImageLayout::eTransferSrcOptimal);

  CopyImageToImage(GetFrame().MainCmdBuffer, current_img, m_DrawImage.Image,
                   m_DrawImage.Extent);

  TransitionImage(GetFrame().MainCmdBuffer, m_DrawImage.Image,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageLayout::eGeneral);

  PostProcess(GetFrame().MainCmdBuffer);

  TransitionImage(GetFrame().MainCmdBuffer, m_DrawImage.Image,
                  vk::ImageLayout::eGeneral,
                  vk::ImageLayout::eTransferSrcOptimal);

  TransitionImage(GetFrame().MainCmdBuffer, current_img,
                  vk::ImageLayout::eTransferSrcOptimal,
                  vk::ImageLayout::eTransferDstOptimal);

  CopyImageToImage(GetFrame().MainCmdBuffer, m_DrawImage.Image, current_img,
                   m_DrawImage.Extent);

  TransitionImage(GetFrame().MainCmdBuffer, current_img,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::ImageLayout::ePresentSrcKHR);

  // close the command buffer
  GetFrame().MainCmdBuffer.end();

  // prepare the submission to the queue.
  vk::CommandBufferSubmitInfo cmd_info;
  cmd_info.setCommandBuffer(GetFrame().MainCmdBuffer);
  cmd_info.setDeviceMask(0);

  vk::SemaphoreSubmitInfo wait_info;
  wait_info.setSemaphore(GetFrame().SwapchainSemaphore);
  wait_info.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
  wait_info.setDeviceIndex(0);
  wait_info.setValue(1);

  vk::SemaphoreSubmitInfo signal_info;
  signal_info.setSemaphore(GetFrame().RenderSemaphore);
  signal_info.setStageMask(vk::PipelineStageFlagBits2::eAllGraphics);
  signal_info.setDeviceIndex(0);
  signal_info.setValue(1);

  vk::SubmitInfo2 submit;
  submit.setWaitSemaphoreInfoCount(1);
  submit.setPWaitSemaphoreInfos(&wait_info);
  submit.setSignalSemaphoreInfoCount(1);
  submit.setPSignalSemaphoreInfos(&signal_info);
  submit.setCommandBufferInfoCount(1);
  submit.setPCommandBufferInfos(&cmd_info);

  VK_CHECK(global.context->GraphicsQueue.submit2(1, &submit,
                                                 GetFrame().RenderFence));

  // present the image
  vk::PresentInfoKHR present_info;
  present_info.setSwapchainCount(1);
  present_info.setPSwapchains(&native_swapchain);
  present_info.setWaitSemaphoreCount(1);
  present_info.setPWaitSemaphores(&GetFrame().RenderSemaphore);
  present_info.setPImageIndices(&swapchain_image_index);

  vk::Result presentResult =
      global.context->GraphicsQueue.presentKHR(&present_info);
  if (presentResult == vk::Result::eErrorOutOfDateKHR) {
    Resize();
    return;
  }

  m_FramesCount++;
}

void Renderer::DrawBackground(vk::CommandBuffer cmd, vk::Image target) {
  vk::ClearColorValue clearValue;
  clearValue.setFloat32({0.1f, 0.1f, 0.1f, 1.0f});

  vk::ImageSubresourceRange clearRange =
      ImageSubresourceRange(vk::ImageAspectFlagBits::eColor);

  cmd.clearColorImage(target, vk::ImageLayout::eGeneral, &clearValue, 1,
                      &clearRange);
}

void Renderer::PostProcess(vk::CommandBuffer cmd) {
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_PostProcessPipeline);

  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                         m_PostProcessPipelineLayout, 0, 1, &GetFrame().ImageSet, 0,
                         nullptr);

  cmd.dispatch(std::ceil(m_DrawImage.Extent.width / 16.0),
               std::ceil(m_DrawImage.Extent.height / 16.0), 1);
}

void Renderer::InitCommands() {
  vk::CommandPoolCreateInfo cmd_pool_info;
  cmd_pool_info.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
  cmd_pool_info.setQueueFamilyIndex(global.context->GraphicsQueueFamily);

  for (std::size_t i = 0; i < FRAME_COUNT; i++) {
    VK_CHECK(global.context->Device.createCommandPool(&cmd_pool_info, nullptr,
                                                      &m_Frames[i].CmdPool));

    vk::CommandBufferAllocateInfo cmd_alloc_info;
    cmd_alloc_info.setCommandPool(m_Frames[i].CmdPool);
    cmd_alloc_info.setCommandBufferCount(1);
    cmd_alloc_info.setLevel(vk::CommandBufferLevel::ePrimary);

    VK_CHECK(global.context->Device.allocateCommandBuffers(
        &cmd_alloc_info, &m_Frames[i].MainCmdBuffer));
  }

  // allocate the command buffer for the immediate mode
  VK_CHECK(global.context->Device.createCommandPool(&cmd_pool_info, nullptr,
                                                    &m_Immediate.CmdPool));

  vk::CommandBufferAllocateInfo cmd_alloc_info;
  cmd_alloc_info.setCommandPool(m_Immediate.CmdPool);
  cmd_alloc_info.setCommandBufferCount(1);
  cmd_alloc_info.setLevel(vk::CommandBufferLevel::ePrimary);

  VK_CHECK(global.context->Device.allocateCommandBuffers(
      &cmd_alloc_info, &m_Immediate.MainCmdBuffer));
}

void Renderer::InitSyncStructures() {
  vk::FenceCreateInfo fence_info;
  fence_info.setFlags(vk::FenceCreateFlagBits::eSignaled);

  vk::SemaphoreCreateInfo semaphore_info;

  for (std::size_t i = 0; i < FRAME_COUNT; i++) {
    VK_CHECK(global.context->Device.createFence(&fence_info, nullptr,
                                                &m_Frames[i].RenderFence));

    VK_CHECK(global.context->Device.createSemaphore(
        &semaphore_info, nullptr, &m_Frames[i].SwapchainSemaphore));
    VK_CHECK(global.context->Device.createSemaphore(
        &semaphore_info, nullptr, &m_Frames[i].RenderSemaphore));
  }

  VK_CHECK(global.context->Device.createFence(&fence_info, nullptr,
                                              &m_Immediate.RenderFence));
}

void Renderer::InitVma() {
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.physicalDevice = global.context->PhysicalDevice;
  allocator_info.device = global.context->Device;
  allocator_info.instance = global.context->Instance;
  allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

  vmaCreateAllocator(&allocator_info, &m_Allocator);

  global.context->DeletionQueue.Push(
      [&]() { vmaDestroyAllocator(m_Allocator); });
}

void Renderer::InitDrawTarget() {
  VkImageUsageFlags usages{};
  usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  usages |= VK_IMAGE_USAGE_STORAGE_BIT;
  usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  m_DrawImage.SetAllocator(m_Allocator, global.context->Device);

  m_DrawImage.Create(global.window->GetSize(), vk::Format::eR16G16B16A16Sfloat,
                     usages);
}

FrameData& Renderer::GetFrame() {
  return m_Frames[m_FramesCount % FRAME_COUNT];
}

void Renderer::ImmediateSubmit(
    std::function<void(VkCommandBuffer cmd)>&& function) {
  VK_CHECK(global.context->Device.resetFences(1, &m_Immediate.RenderFence));
  m_Immediate.MainCmdBuffer.reset();

  vk::CommandBuffer cmd = m_Immediate.MainCmdBuffer;
  vk::CommandBufferBeginInfo cmd_begin_info;
  cmd_begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  VK_CHECK(cmd.begin(&cmd_begin_info));

  function(cmd);

  cmd.end();

  vk::CommandBufferSubmitInfo cmd_info;
  cmd_info.setCommandBuffer(cmd);
  cmd_info.setDeviceMask(0);

  vk::SubmitInfo2 submit;
  submit.setCommandBufferInfoCount(1);
  submit.setPCommandBufferInfos(&cmd_info);

  VK_CHECK(global.context->GraphicsQueue.submit2(1, &submit,
                                                 m_Immediate.RenderFence));

  VK_CHECK(global.context->Device.waitForFences(1, &m_Immediate.RenderFence,
                                                true, 9999999999));
}

void Renderer::DrawImGui(vk::CommandBuffer cmd, vk::ImageView target) {
  vk::RenderingAttachmentInfo color_attachment;
  color_attachment.setImageView(target);
  color_attachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
  color_attachment.setLoadOp(vk::AttachmentLoadOp::eLoad);
  color_attachment.setStoreOp(vk::AttachmentStoreOp::eStore);

  vk::Rect2D rect;
  rect.setExtent({m_DrawImage.Extent.width, m_DrawImage.Extent.height});

  vk::RenderingInfo render_info;
  render_info.setRenderArea(rect);
  render_info.setLayerCount(1);
  render_info.setViewMask(0);
  render_info.setColorAttachmentCount(1);
  render_info.setPColorAttachments(&color_attachment);

  cmd.beginRendering(&render_info);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  cmd.endRendering();
}

void Renderer::InitPipelines() {
  vk::PipelineLayoutCreateInfo compute_info;
  compute_info.setSetLayoutCount(1);
  compute_info.setPSetLayouts(&m_DrawImageDescriptorLayout);

  VK_CHECK(global.context->Device.createPipelineLayout(
      &compute_info, nullptr, &m_PostProcessPipelineLayout));

  vk::ShaderModule compute_draw_shader;
  LoadShaderModule("../res/shaders/compiled/postprocess.spv",
                   global.context->Device, &compute_draw_shader);

  vk::PipelineShaderStageCreateInfo stage_info;
  stage_info.setStage(vk::ShaderStageFlagBits::eCompute);
  stage_info.setModule(compute_draw_shader);
  stage_info.setPName("main");

  vk::ComputePipelineCreateInfo compute_pipeline_info;
  compute_pipeline_info.setLayout(m_PostProcessPipelineLayout);
  compute_pipeline_info.setStage(stage_info);

  VK_CHECK(global.context->Device.createComputePipelines(
      VK_NULL_HANDLE, 1, &compute_pipeline_info, nullptr,
      &m_PostProcessPipeline));

  global.context->Device.destroyShaderModule(compute_draw_shader);

  global.context->DeletionQueue.Push([&]() {
    global.context->Device.destroyPipelineLayout(m_PostProcessPipelineLayout);
    global.context->Device.destroyPipeline(m_PostProcessPipeline);
  });
}

void Renderer::InitDescriptors() {
  DescriptorLayoutBuilder builder;
  builder.add_binding(0, vk::DescriptorType::eStorageImage);
  m_DrawImageDescriptorLayout =
      builder.build(global.context->Device, vk::ShaderStageFlagBits::eCompute);

  for (int i = 0; i < FRAME_COUNT; i++) {
    std::vector<PoolSizeRatio> frame_sizes = {
        {vk::DescriptorType::eStorageImage, 3},
        {vk::DescriptorType::eStorageBuffer, 3},
        {vk::DescriptorType::eUniformBuffer, 3},
        {vk::DescriptorType::eCombinedImageSampler, 4},
    };

    m_Frames[i].Descriptors.Init(global.context->Device, 1000, frame_sizes);

    m_Frames[i].ImageSet = m_Frames[i].Descriptors.Allocate(m_DrawImageDescriptorLayout);

    DescriptorWriter writer;
    writer.WriteImage(0, m_DrawImage.View, VK_NULL_HANDLE,
                      vk::ImageLayout::eGeneral,
                      vk::DescriptorType::eStorageImage);

    writer.UpdateSet(global.context->Device, m_Frames[i].ImageSet);

    m_Frames[i].DeletionQueue.Push(
        [&, i]() { m_Frames[i].Descriptors.DestroyPools(); });
  }
}

void Renderer::Resize() {
  global.context->Device.waitIdle();

  global.swapchain->Shutdown();
  global.swapchain->Init();

  m_DrawImage.Destroy();

  InitDrawTarget();

  for (int i = 0; i < FRAME_COUNT; i++) {
    DescriptorWriter writer;
    writer.WriteImage(0, m_DrawImage.View, VK_NULL_HANDLE,
                      vk::ImageLayout::eGeneral,
                      vk::DescriptorType::eStorageImage);

    writer.UpdateSet(global.context->Device, m_Frames[i].ImageSet);
  }
}

float Renderer::GetFPS() const {
  auto current_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                      current_time - m_StartTime)
                      .count();
  return (float)m_FramesCount / duration;
}

}  // namespace gfx
