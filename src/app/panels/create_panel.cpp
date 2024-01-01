//
// Created by Stefan on 1/1/2024.
//

#include "app/panels/create_panel.hpp"

#include <imgui.h>

namespace DiffusionStudio {

CreatePanel::CreatePanel() {}

CreatePanel::~CreatePanel() {}

void CreatePanel::Draw() {
  ImGui::Begin("Control Panel");

  if (ImGui::Button("Imagine")) {
  }

  ImGui::End();
}

}  // namespace DiffusionStudio
