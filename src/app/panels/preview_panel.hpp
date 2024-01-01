//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "app/panels/panel.hpp"

#include "gfx/allocated_image.hpp"

namespace DiffusionStudio {

class PreviewPanel : public Panel {
 public:
  PreviewPanel();
  ~PreviewPanel() override;

  void SetPreviewImage(vk::DescriptorSet img);

  void Draw() override;

 private:
  void *m_PreviewImage;
};

}  // namespace DiffusionStudio
