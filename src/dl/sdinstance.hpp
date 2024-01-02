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
  StableDiffusionInstance();
  ~StableDiffusionInstance();

  void AddModel(const std::function<void()>& callback);

  void Generate(const std::string& prompt, const std::function<void(uint8_t*)>& callback);

 private:
  void DiffusionThread(util::ThreadSafeQueue<std::function<void()>>& tasks) const;

 private:
  std::thread m_DiffusionThread;
  util::ThreadSafeQueue<std::function<void()>> m_Tasks;
  bool m_Working = true;

  std::unordered_map<std::string, std::shared_ptr<DiffusionModel>> m_Models;
};

}  // namespace dl
