#include "global.hpp"

int main() {
  init_globals();

  while (!global.window->ShouldClose()) {
    global.window->PollEvents();

    global.imgui->AddPanel([&]() {
      ImGui::Begin("Control Panel");

      ImGui::End();
    });

    global.imgui->Draw();
    global.renderer->Draw();
  }

  return 0;
}
