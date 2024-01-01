//
// Created by Stefan on 1/1/2024.
//

#include "dl/diffusionmodel.hpp"

#include "util/util.hpp"

namespace dl {

DiffusionModel::DiffusionModel(const DiffusionModelProps& props) {
  util::error::ErrNDie(!m_TempSD.load_from_file(props.Checkpoint.string(), "", GGML_TYPE_F16, Schedule::DEFAULT), "Failed to load SD model.");
}

uint8_t* DiffusionModel::Generate(const std::string& prompt) {
  std::vector<uint8_t*> batch = m_TempSD.txt2img(prompt, "", 7.0f, 512, 512, SampleMethod::EULER_A, 20, 69420, 1);
  return batch[0];
}

DiffusionModel::~DiffusionModel() = default;

}
