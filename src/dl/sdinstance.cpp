//
// Created by Stefan on 1/2/2024.
//

#include "dl/sdinstance.hpp"

namespace dl {

StableDiffusionInstance::StableDiffusionInstance() {
  m_DiffusionThread =
      std::thread([this]() { this->DiffusionThread(std::ref(m_Tasks)); });
}

StableDiffusionInstance::~StableDiffusionInstance() {
  if (m_DiffusionThread.joinable()) {
    m_Working = false;
    m_DiffusionThread.join();
  }
}

// TODO
void StableDiffusionInstance::AddModel(const std::function<void()>& callback) {
  m_Tasks.Push([&, callback]() {
    m_Models["a"] = std::make_shared<DiffusionModel>(DiffusionModelProps{
        .Checkpoint =
            "D:\\StableDiffusion\\stable-diffusion-webui\\models\\Stable-"
            "diffusion\\v1-5-pruned-emaonly.safetensors",
        .Loras = "",
        .VariationalAutoencoder = ""});

    callback();
  });
}

void StableDiffusionInstance::DiffusionThread(
    util::ThreadSafeQueue<std::function<void()>>& tasks) const {
  while (m_Working) {
    // todo: use a cv
    std::function<void()> task;
    if (tasks.TryPop(task)) {
      task();
    }
  }
}

void StableDiffusionInstance::Generate(
    const std::string& prompt, const std::function<void(uint8_t*)>& callback) {
  m_Tasks.Push([=, this]() {
    for (uint8_t* sample :
         m_Models["a"]->Generate(prompt) | std::views::take(20)) {
      callback(sample);
    }
  });
}

}  // namespace dl
