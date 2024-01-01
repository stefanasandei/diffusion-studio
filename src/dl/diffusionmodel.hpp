//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "util/util.hpp"

namespace dl {

struct DiffusionModelProps {
  std::filesystem::path Checkpoint;
  std::filesystem::path Loras;
  std::filesystem::path VariationalAutoencoder;
};

class DiffusionModel {
 public:
  DiffusionModel(const DiffusionModelProps& props) {}
  ~DiffusionModel() {}
};

}
