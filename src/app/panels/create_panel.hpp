//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "app/panels/panel.hpp"

namespace DiffusionStudio {

class CreatePanel : public Panel {
 public:
  CreatePanel();
  ~CreatePanel() override;

  void Draw() override;
};

}  // namespace DiffusionStudio
