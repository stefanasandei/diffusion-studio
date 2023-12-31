//
// Created by Stefan on 12/26/2023.
//

#include "gfx/shader.hpp"

#include <iostream>
#include <fstream>

namespace gfx {

bool LoadShaderModule(const char* filepath, vk::Device device,
                        vk::ShaderModule* module) {
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) return false;

  std::size_t filesize = file.tellg();
  std::vector<uint32_t> buffer(filesize / sizeof(uint32_t));

  file.seekg(0);
  file.read((char*)buffer.data(), filesize);
  file.close();

  vk::ShaderModuleCreateInfo info;
  info.setCodeSize(buffer.size() * sizeof(uint32_t));
  info.setPCode(buffer.data());

  vk::ShaderModule shader_module;
  VK_CHECK(device.createShaderModule(&info, nullptr, &shader_module));

  *module = shader_module;
  return true;
}

}  // namespace gfx
