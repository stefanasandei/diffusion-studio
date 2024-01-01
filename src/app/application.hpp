//
// Created by Stefan on 12/31/2023.
//

#pragma once

#include "util/std.hpp"

#include "app/panels/panel.hpp"
#include "app/panels/create_panel.hpp"

namespace DiffusionStudio {

class Application {
 public:
  explicit Application(std::span<std::string_view> args);
  ~Application();

  int32_t Run();

 private:
  void DrawDockingSpace() const;
  void DrawMainMenu();

 private:
  CreatePanel m_CreatePanel;

  bool m_IsRunning = true;
};

}  // namespace DiffusionStudio
