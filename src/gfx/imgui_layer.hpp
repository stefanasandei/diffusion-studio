//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "util/std.hpp"

#include "gfx/allocated_image.hpp"

#include <imgui.h>

namespace gfx {

class ImGUILayer {
 public:
  ImGUILayer();
  ~ImGUILayer();

  void AddPanel(const std::function<void()>& draw_fn);
  void Draw();

  static vk::DescriptorSet UploadImage(const AllocatedImage& img);

 private:
  void init();

 private:
  std::vector<std::function<void()>> m_Panels;

  vk::DescriptorPool m_Pool;
};

}  // namespace gfx