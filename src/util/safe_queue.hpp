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
    {
      std::unique_lock<std::mutex> lock(m_ItemsMutex);
      m_Items.push(item);
    }
    m_ItemAdded.notify_all();
  }

  bool TryPop(T& item) {
    std::lock_guard<std::mutex> lock(m_ItemsMutex);

    if (m_Items.empty()) {
      return false;
    }

    item = m_Items.front();
    m_Items.pop();
    return true;
  }

  [[nodiscard]] std::size_t GetSize() { return m_Items.size(); }

  [[nodiscard]] std::condition_variable& GetItemAdded() { return m_ItemAdded; }
  [[nodiscard]] std::mutex& GetItemsMutex() { return m_ItemsMutex; }

 private:
  std::queue<T> m_Items;
  std::mutex m_ItemsMutex;
  std::condition_variable m_ItemAdded;
};

}  // namespace util
