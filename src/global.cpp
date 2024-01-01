//
// Created by Stefan on 12/24/2023.
//

#include "global.hpp"

static Global global_state;
Global& global = global_state;

void init_globals() {
  global.window =
      std::make_unique<platform::Window>(glm::ivec2(1600, 900), "Diffusion Studio");

  global.context = std::make_unique<gfx::Context>();

  global.swapchain = std::make_unique<gfx::Swapchain>();

  global.renderer = std::make_unique<gfx::Renderer>();

  global.imgui = std::make_unique<gfx::ImGUILayer>();

  global.sd = std::make_unique<dl::StableDiffusionInstance>();
}

