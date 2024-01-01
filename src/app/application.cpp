//
// Created by Stefan on 12/31/2023.
//

#include "app/application.hpp"

#include "global.hpp"

namespace DiffusionStudio {


Application::Application(std::span<std::string_view> args) {
  init_globals();
}

Application::~Application() = default;

int32_t Application::Run() {
  while (!global.window->ShouldClose()) {
    global.window->PollEvents();

    DrawDockingSpace();
    DrawMainMenu();

    global.imgui->AddPanel([&]() {
      ImGui::Begin("Control Panel");

      if (ImGui::Button("Imagine")) {
      }

      ImGui::End();
    });

    global.imgui->Draw();
    global.renderer->Draw();
  }

  return 0;
}

void Application::DrawMainMenu() const {
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
