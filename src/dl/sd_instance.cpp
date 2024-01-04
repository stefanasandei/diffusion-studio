//
// Created by Stefan on 1/2/2024.
//

#include "dl/sd_instance.hpp"

#include <ggml/ggml.h>

namespace dl {

StableDiffusionInstance::StableDiffusionInstance() {
  m_DiffusionThread =
      std::thread([this]() { this->DiffusionThread(std::ref(m_Tasks)); });
}

StableDiffusionInstance::~StableDiffusionInstance() {
  if (m_DiffusionThread.joinable()) {
    m_Working = false;
    m_Tasks.GetItemAdded().notify_all();
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
    util::ThreadSafeQueue<std::function<void()>>& tasks) {
  while (m_Working) {
    {
      std::unique_lock<std::mutex> lock(const_cast<std::mutex&>(m_Tasks.GetItemsMutex()));
      m_Tasks.GetItemAdded().wait(lock, [&, this]() {
        return !m_Working || tasks.GetSize() > 0;
      });
    }

    std::function<void()> task;
    if (tasks.TryPop(task)) {
      task();
    }
  }
}

void StableDiffusionInstance::Generate(
    const std::string& prompt, const std::function<void(uint8_t*)>& callback) {
  m_Tasks.Push([=, this]() {
    int steps = 20;

    for (ggml_tensor* sample :
         m_Models["a"]->Generate(prompt) | std::views::take(steps + 1)) {

      std::future<void> res = std::async(std::launch::async, [=, this]() {
        std::lock_guard<std::mutex> lock(m_DecodingMutex);

        uint8_t* image = m_Models["a"]->ExtractSample(sample);
        callback(image);
      });

      m_PendingFutures.push_back(std::move(res)); // TODO not really fully async
    }
  });
}

}  // namespace dl
