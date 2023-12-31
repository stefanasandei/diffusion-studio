//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"

namespace gfx {

bool LoadShaderModule(const char* filepath, vk::Device device,
                      vk::ShaderModule* module);

}