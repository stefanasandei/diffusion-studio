//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "util/util.hpp"

#include <stable-diffusion.h>

namespace dl {

struct DiffusionModelProps {
  std::filesystem::path Checkpoint;
  std::filesystem::path Loras;
  std::filesystem::path VariationalAutoencoder;
};

class DiffusionModel {
 public:
  DiffusionModel() = default;
  explicit DiffusionModel(const DiffusionModelProps& props);
  ~DiffusionModel();

  std::experimental::generator<uint8_t*> Generate(std::string prompt);

 private:
  StableDiffusion m_TempSD;
};

}  // namespace dl
