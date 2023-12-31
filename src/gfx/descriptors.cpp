//
// Created by Stefan on 12/28/2023.
//

#include "gfx/descriptors.hpp"

#include <iostream>

namespace gfx {

DescriptorAllocator::DescriptorAllocator() {}

DescriptorAllocator::~DescriptorAllocator() {}

void DescriptorAllocator::Init(vk::Device device, uint32_t initialSets,
                               std::span<PoolSizeRatio> poolRatios) {
  this->device = device;

  ratios.clear();

  for (auto r : poolRatios) {
    ratios.push_back(r);
  }

  vk::DescriptorPool newPool = CreatePool(initialSets, poolRatios);

  setsPerPool = initialSets * 1.5;

  readyPools.push_back(newPool);
}

void DescriptorAllocator::ClearPools() {
  for (auto p : readyPools) {
    vkResetDescriptorPool(device, p, 0);
  }
  for (auto p : fullPools) {
    vkResetDescriptorPool(device, p, 0);
    readyPools.push_back(p);
  }
  fullPools.clear();
}

void DescriptorAllocator::DestroyPools() {
  for (auto p : readyPools) {
    vkDestroyDescriptorPool(device, p, nullptr);
  }
  readyPools.clear();
  for (auto p : fullPools) {
    vkDestroyDescriptorPool(device, p, nullptr);
  }
  fullPools.clear();
}

vk::DescriptorSet DescriptorAllocator::Allocate(
    vk::DescriptorSetLayout layout) {
  vk::DescriptorPool poolToUse = GetPool();

  vk::DescriptorSetAllocateInfo allocateInfo;
  allocateInfo.setDescriptorPool(poolToUse);
  allocateInfo.setDescriptorSetCount(1);
  allocateInfo.setPSetLayouts(&layout);

  vk::DescriptorSet ds;
  vk::Result res = device.allocateDescriptorSets(&allocateInfo, &ds);

  if (res == vk::Result::eErrorOutOfPoolMemory ||
      res == vk::Result::eErrorFragmentedPool) {
    fullPools.push_back(poolToUse);

    poolToUse = GetPool();
    allocateInfo.setDescriptorPool(poolToUse);

    VK_CHECK(device.allocateDescriptorSets(&allocateInfo, &ds));
  }

  return ds;
}

vk::DescriptorPool DescriptorAllocator::GetPool() {
  vk::DescriptorPool newPool;
  if (!readyPools.empty()) {
    newPool = readyPools.back();
    readyPools.pop_back();
  } else {
    newPool = CreatePool(setsPerPool, ratios);

    setsPerPool = static_cast<int32_t>(setsPerPool * 1.5);
    if (setsPerPool > 4092) {
      setsPerPool = 4092;
    }
  }

  return newPool;
}

vk::DescriptorPool DescriptorAllocator::CreatePool(
    uint32_t setCount, std::span<PoolSizeRatio> poolRatios) {
  std::vector<vk::DescriptorPoolSize> poolSizes;
  for (PoolSizeRatio ratio : poolRatios) {
    vk::DescriptorPoolSize size;
    size.setType(ratio.type);
    size.setDescriptorCount(static_cast<uint32_t>(ratio.ratio * setCount));

    poolSizes.push_back(size);
  }

  vk::DescriptorPoolCreateInfo info;
  info.setMaxSets(setCount);
  info.setPoolSizeCount(poolSizes.size());
  info.setPPoolSizes(poolSizes.data());

  vk::DescriptorPool newPool;
  VK_CHECK(device.createDescriptorPool(&info, nullptr, &newPool));

  return newPool;
}

DescriptorWriter::DescriptorWriter() {}

DescriptorWriter::~DescriptorWriter() {}

void DescriptorWriter::WriteImage(int binding, vk::ImageView image,
                                  vk::Sampler sampler, vk::ImageLayout layout,
                                  vk::DescriptorType type) {
  vk::DescriptorImageInfo info;
  info.setSampler(sampler);
  info.setImageView(image);
  info.setImageLayout(layout);

  imageInfos.emplace_back(info);

  vk::WriteDescriptorSet write;
  write.setDstBinding(binding);
  write.setDstSet(VK_NULL_HANDLE);
  write.setDescriptorCount(1);
  write.setDescriptorType(type);
  write.setPImageInfo(&info);

  writes.push_back(write);
}

void DescriptorWriter::WriteBuffer(int binding, vk::Buffer buffer, size_t size,
                                   size_t offset, vk::DescriptorType type) {
  vk::DescriptorBufferInfo info;
  info.setBuffer(buffer);
  info.setOffset(offset);
  info.setRange(size);

  bufferInfos.emplace_back(info);

  vk::WriteDescriptorSet write;
  write.setDstBinding(binding);
  write.setDstSet(VK_NULL_HANDLE);
  write.setDescriptorCount(1);
  write.setDescriptorType(type);
  write.setPBufferInfo(&info);

  writes.push_back(write);
}

void DescriptorWriter::Clear() {
  imageInfos.clear();
  writes.clear();
  bufferInfos.clear();
}

void DescriptorWriter::UpdateSet(vk::Device device, vk::DescriptorSet set) {
  for (vk::WriteDescriptorSet& write : writes) {
    write.setDstSet(set);
  }

  device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);
}

DescriptorLayoutBuilder::DescriptorLayoutBuilder() {}

DescriptorLayoutBuilder::~DescriptorLayoutBuilder() {}

void DescriptorLayoutBuilder::add_binding(uint32_t binding,
                                          vk::DescriptorType type) {
  vk::DescriptorSetLayoutBinding newbind;
  newbind.setBinding(binding);
  newbind.setDescriptorCount(1);
  newbind.setDescriptorType(type);

  m_Bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear() { m_Bindings.clear(); }

vk::DescriptorSetLayout DescriptorLayoutBuilder::build(
    vk::Device device, vk::ShaderStageFlags shader_stages) {
  for (auto& binding : m_Bindings) {
    binding.setStageFlags(binding.stageFlags | shader_stages);
  }

  vk::DescriptorSetLayoutCreateInfo info;
  info.setBindingCount(m_Bindings.size());
  info.setPBindings(m_Bindings.data());

  vk::DescriptorSetLayout set;
  VK_CHECK(device.createDescriptorSetLayout(&info, nullptr, &set));

  return set;
}

}  // namespace gfx
