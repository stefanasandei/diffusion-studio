//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "util/std.hpp"

#include <imgui.h>

namespace gfx {

class ImGUILayer {
 public:
  ImGUILayer();
  ~ImGUILayer();

  void AddPanel(const std::function<void()>& draw_fn);
  void Draw();

 private:
  void init();

 private:
  std::vector<std::function<void()>> m_Panels;
};

}  // namespace gfx