//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "dl/diffusionmodel.hpp"
#include "util/util.hpp"

#include <experimental/generator>

namespace dl {

class StableDiffusionInstance {
 public:
  StableDiffusionInstance() {}
  ~StableDiffusionInstance() {}

  void AddModel() {}
  std::experimental::generator<uint8_t*> Generate() {}

 private:
  std::unordered_map<std::string, DiffusionModel> m_Models;
};

}  // namespace dl
