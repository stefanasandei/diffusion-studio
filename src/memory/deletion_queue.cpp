//
// Created by Stefan on 12/26/2023.
//

#include "memory/deletion_queue.hpp"

namespace memory {

DeletionQueue::DeletionQueue() {}

DeletionQueue::~DeletionQueue() {}

void DeletionQueue::Push(std::function<void()>&& function) {
  m_Deletors.push_back(function);
}

void DeletionQueue::Flush() {
  for (auto it = m_Deletors.rbegin(); it != m_Deletors.rend(); it++) {
    (*it)();
  }

  m_Deletors.clear();
}

}  // namespace memory
