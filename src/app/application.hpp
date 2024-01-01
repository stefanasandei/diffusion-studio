//
// Created by Stefan on 12/31/2023.
//

#pragma once

#include "util/std.hpp"

namespace DiffusionStudio {

class Application {
 public:
  explicit Application(std::span<std::string_view> args);
  ~Application();

  int32_t Run();

 private:
  void DrawDockingSpace() const;
  void DrawMainMenu() const;
};

}  // namespace
