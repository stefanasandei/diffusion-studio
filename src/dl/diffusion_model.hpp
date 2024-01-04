//
// Created by Stefan on 1/1/2024.
//

#pragma once

#include "util/util.hpp"

class StableDiffusionGGML;
enum SampleMethod;
struct ggml_tensor;
struct ggml_context;

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

  std::experimental::generator<ggml_tensor*> Generate(std::string prompt);
  uint8_t* ExtractSample(ggml_tensor* sample);

 private:
  ggml_tensor* Decode(ggml_context* work_ctx, ggml_tensor* input);
  uint8_t* Upscale(ggml_tensor* image);

  std::experimental::generator<ggml_tensor*> Sample(
      ggml_context* work_context, ggml_tensor* x_t, ggml_tensor* noise,
      ggml_tensor* c, ggml_tensor* c_vector, ggml_tensor* uc,
      ggml_tensor* uc_vector, float cfg_scale, SampleMethod method,
      const std::vector<float>& sigmas);

 private:
  std::shared_ptr<StableDiffusionGGML> m_StableDiffusionGGMLBackend;

  ggml_tensor* m_SampleCopy;
};

}  // namespace dl
