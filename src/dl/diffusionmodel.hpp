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
  explicit DiffusionModel(const DiffusionModelProps& props);
  ~DiffusionModel();

  uint8_t* Generate(const std::string& prompt);

 private:
  StableDiffusion m_TempSD;
};

}  // namespace dl
