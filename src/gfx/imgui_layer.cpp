//
// Created by Stefan on 12/26/2023.
//

#include "gfx/imgui_layer.hpp"

#include "global.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace gfx {

ImGUILayer::ImGUILayer() { init(); }

ImGUILayer::~ImGUILayer() = default;

void ImGUILayer::init() {
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  vk::DescriptorPoolCreateInfo pool_info;
  pool_info.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
  pool_info.setMaxSets(1000);
  pool_info.setPoolSizeCount(std::size(pool_sizes));
  pool_info.setPPoolSizes(
      reinterpret_cast<const vk::DescriptorPoolSize*>(pool_sizes));

  VK_CHECK(global.context->Device.createDescriptorPool(&pool_info, nullptr,
                                                       &m_Pool));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui_ImplGlfw_InitForVulkan(global.window->GetNative(), true);

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = global.context->Instance;
  init_info.PhysicalDevice = global.context->PhysicalDevice;
  init_info.Device = global.context->Device;
  init_info.Queue = global.context->GraphicsQueue;
  init_info.DescriptorPool = m_Pool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = true;
  init_info.ColorAttachmentFormat = global.swapchain->ImageFormat;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info, VK_NULL_HANDLE);

  global.renderer->ImmediateSubmit(
      [&](VkCommandBuffer cmd) { ImGui_ImplVulkan_CreateFontsTexture(); });

  global.context->DeletionQueue.Push([&]() { ImGui_ImplVulkan_Shutdown(); });
}

void ImGUILayer::AddPanel(const std::function<void()>& draw_fn) {
  m_Panels.push_back(draw_fn);
}

void ImGUILayer::Draw() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  for (const auto& fn : m_Panels) fn();
  m_Panels.clear();

  ImGui::Render();
}

vk::DescriptorSet ImGUILayer::UploadImage(const AllocatedImage& img) {
  vk::SamplerCreateInfo info;
  info.setMinFilter(vk::Filter::eLinear);
  info.setMagFilter(vk::Filter::eLinear);
  vk::Sampler sampler;

  VK_CHECK(global.context->Device.createSampler(&info, nullptr, &sampler));

  vk::DescriptorSet set = ImGui_ImplVulkan_AddTexture(sampler, img.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  return set;
}

}  // namespace gfx
