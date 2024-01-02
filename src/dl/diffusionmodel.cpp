//
// Created by Stefan on 1/1/2024.
//

#include "dl/diffusionmodel.hpp"

#include "util/util.hpp"

namespace dl {

DiffusionModel::DiffusionModel(const DiffusionModelProps& props) {
  util::error::ErrNDie(!m_TempSD.load_from_file(props.Checkpoint.string(), "", GGML_TYPE_F16, Schedule::DEFAULT), "Failed to load SD model.");
}

std::experimental::generator<uint8_t*> DiffusionModel::Generate(std::string prompt) {
  std::vector<uint8_t*> batch = m_TempSD.txt2img(prompt.c_str(), "", 7.0f, 512, 512, SampleMethod::EULER_A, 20, 69420, 1);
  auto* real_res = new uint8_t[512 * 512 * 4];
  for (int i = 0; i < 512 * 512; ++i) {
    real_res[i * 4] = batch[0][i * 3];       // Copy Red
    real_res[i * 4 + 1] = batch[0][i * 3 + 1]; // Copy Green
    real_res[i * 4 + 2] = batch[0][i * 3 + 2]; // Copy Blue
    real_res[i * 4 + 3] = 255;  // Set Alpha to 255 (fully opaque)
  }

  co_yield real_res;
}

DiffusionModel::~DiffusionModel() = default;

}
