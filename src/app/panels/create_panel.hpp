//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "app/panels/panel.hpp"

#include "util/util.hpp"

namespace DiffusionStudio {

class CreatePanel : public Panel {
 public:
  CreatePanel();
  ~CreatePanel() override;

  void Draw() override;

  void SetGenerationCallback(const std::function<void(uint8_t*)>& callback);

 private:
  std::string m_Prompt;
  std::function<void(uint8_t*)> m_GenerationCallback;

  int m_Loaded = 0;
};

}  // namespace DiffusionStudio
