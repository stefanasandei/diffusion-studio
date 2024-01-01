//
// Created by Stefan on 1/1/2024.
//

#pragma once

namespace DiffusionStudio {

class Panel {
 public:
  virtual ~Panel() = default;

  virtual void Draw() = 0;
};

}  // namespace DiffusionStudio
