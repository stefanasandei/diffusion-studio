//
// Created by Stefan on 12/24/2023.
//

#pragma once

#include "platform/window.h"

#include "gfx/context.hpp"
#include "gfx/swapchain.hpp"
#include "gfx/renderer.hpp"
#include "gfx/imgui_layer.hpp"

#include "dl/sdinstance.hpp"

void init_globals();

struct Global {
  std::unique_ptr<platform::Window> window;

  std::unique_ptr<gfx::Context> context;
  std::unique_ptr<gfx::Swapchain> swapchain;
  std::unique_ptr<gfx::Renderer> renderer;
  std::unique_ptr<gfx::ImGUILayer> imgui;

  std::unique_ptr<dl::StableDiffusionInstance> sd;
};

extern Global& global;
