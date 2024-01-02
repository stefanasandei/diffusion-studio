//
// Created by Stefan on 12/31/2023.
//

#include "app/application.hpp"

#include "global.hpp"

#include "gfx/image.hpp"

#include <stb_image.h>

namespace DiffusionStudio {

Application::Application(std::span<std::string_view> args) {
  init_globals();

  // test usage only
  int width, height, comp;
  uint8_t* data = stbi_load("output-8.png", &width, &height, &comp, 4);

  gfx::AllocatedImage img = gfx::CreateImage(data, {width, height}, vk::Format::eR8G8B8A8Unorm, VK_IMAGE_USAGE_SAMPLED_BIT);

  auto imageSet = global.imgui->UploadImage(img);

  m_PreviewPanel.SetPreviewImage(imageSet);

  m_CreatePanel.SetGenerationCallback([=](uint8_t* data) {
    UpdateImageData(data, img);
  });
}

Application::~Application() = default;

int32_t Application::Run() {
  while (!global.window->ShouldClose() && m_IsRunning) {
    global.window->PollEvents();

    DrawDockingSpace();
    DrawMainMenu();

    global.imgui->AddPanel([&]() { m_CreatePanel.Draw(); });
    global.imgui->AddPanel([&]() { m_PreviewPanel.Draw(); });

    global.imgui->Draw();
    global.renderer->Draw();
  }

  return 0;
}

void Application::DrawMainMenu() {
  global.imgui->AddPanel([&]() {
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New")) {
        }
        if (ImGui::MenuItem("Open")) {
        }
        if (ImGui::MenuItem("Save")) {
        }
        if (ImGui::MenuItem("Settings")) {
        }
        if (ImGui::MenuItem("Exit")) {
          m_IsRunning = false;
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("User Guide")) {
        }
        if (ImGui::MenuItem("About")) {
        }
        ImGui::EndMenu();
      }
    }
    ImGui::EndMainMenuBar();
  });
}

void Application::DrawDockingSpace() const {
  global.imgui->AddPanel([&]() {
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
      window_flags |= ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("###DockSpace", nullptr, window_flags);

    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);

    ImGuiIO& io = ImGui::GetIO();
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    ImGui::End();
  });
}

}  // namespace DiffusionStudio
