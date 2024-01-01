//
// Created by Stefan on 1/1/2024.
//

#include "app/panels/preview_panel.hpp"

#include <imgui.h>

namespace DiffusionStudio {

PreviewPanel::PreviewPanel() { m_PreviewImage = nullptr; }

PreviewPanel::~PreviewPanel() {}

void PreviewPanel::Draw() {
  ImGui::Begin("Preview");

  if (m_PreviewImage != nullptr) {
    ImGui::Image(
        static_cast<ImTextureID>(m_PreviewImage),
        ImVec2(512, 512));
  }

  ImGui::End();
}
void PreviewPanel::SetPreviewImage(
    vk::DescriptorSet img) {
  m_PreviewImage = img;
}

}  // namespace DiffusionStudio
