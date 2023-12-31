//
// Created by Stefan on 12/28/2023.
//

#pragma once

#include "gfx/vulkan_utils.hpp"

#include <deque>

namespace gfx {

struct PoolSizeRatio {
  vk::DescriptorType type;
  float ratio;
};

class DescriptorAllocator {
 public:
  DescriptorAllocator();
  ~DescriptorAllocator();

  void Init(vk::Device device, uint32_t initialSets,
            std::span<PoolSizeRatio> poolRatios);
  void ClearPools();
  void DestroyPools();

  vk::DescriptorSet Allocate(vk::DescriptorSetLayout layout);

 private:
  vk::DescriptorPool GetPool();
  vk::DescriptorPool CreatePool(uint32_t setCount,
                                std::span<PoolSizeRatio> poolRatios);

 private:
  std::vector<PoolSizeRatio> ratios;
  std::vector<vk::DescriptorPool> fullPools;
  std::vector<vk::DescriptorPool> readyPools;
  uint32_t setsPerPool;

  vk::Device device;
};

class DescriptorLayoutBuilder {
 public:
  DescriptorLayoutBuilder();
  ~DescriptorLayoutBuilder();

  void add_binding(uint32_t binding, vk::DescriptorType type);
  void clear();

  vk::DescriptorSetLayout build(vk::Device device,
                                vk::ShaderStageFlags shader_stages);

 private:
  std::vector<vk::DescriptorSetLayoutBinding> m_Bindings;
};

class DescriptorWriter {
 public:
  DescriptorWriter();
  ~DescriptorWriter();

  void WriteImage(int binding, vk::ImageView image, vk::Sampler sampler, vk::ImageLayout layout, vk::DescriptorType type);
  void WriteBuffer(int binding, vk::Buffer buffer, size_t size, size_t offset, vk::DescriptorType type);

  void Clear();
  void UpdateSet(vk::Device device, vk::DescriptorSet set);

 private:
  std::deque<vk::DescriptorImageInfo> imageInfos;
  std::deque<vk::DescriptorBufferInfo> bufferInfos;
  std::vector<vk::WriteDescriptorSet> writes;
};

}  // namespace gfx