//
// Created by Stefan on 1/2/2024.
//

#pragma once

#include "util/std.hpp"

namespace util {

template <typename T>
class ThreadSafeQueue {
 public:
  void Push(const T& item) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Items.push(item);
  }

  bool TryPop(T& item) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_Items.empty()) {
      return false;
    }

    item = m_Items.front();
    m_Items.pop();
    return true;
  }

 private:
  std::queue<T> m_Items;
  std::mutex m_Mutex;
};

}  // namespace util
