//
// Created by Stefan on 1/1/2024.
//

#include "app/panels/create_panel.hpp"

#include "global.hpp"

#include <imgui.h>
#include "gfx/imgui_stdlib.hpp"

namespace DiffusionStudio {

CreatePanel::CreatePanel() {
}

CreatePanel::~CreatePanel() {}

void CreatePanel::Draw() {
  ImGui::Begin("Control Panel");

  // TODO: fix this model loading workflow
  if(m_Loaded == 2) {
    ImGui::InputText("Prompt", &m_Prompt);

    if (ImGui::Button("Imagine")) {
      global.sd->Generate(
          m_Prompt, [&](uint8_t* sample) { m_GenerationCallback(sample); });
    }
  }

  if(m_Loaded == 0 && ImGui::Button("Load Model")) {
    m_Loaded = 1;
    global.sd->AddModel([&]() {
      m_Loaded = 2;
    });
  }

  if(m_Loaded == 1) {
    ImGui::Text("Loading...");
  } else if(m_Loaded == 2) {
    ImGui::Text("Model Loaded");
  }

  ImGui::End();
}
void CreatePanel::SetGenerationCallback(
    const std::function<void(uint8_t *)>& callback) {
  m_GenerationCallback = callback;
}

}  // namespace DiffusionStudio
