//
// Created by Stefan on 12/26/2023.
//

#pragma once

#include "util/std.hpp"

namespace memory {

class DeletionQueue {
 public:
  DeletionQueue();
  ~DeletionQueue();

  void Push(std::function<void()>&& function);
  void Flush();

 private:
  std::deque<std::function<void()>> m_Deletors;
};

}  // namespace memory
