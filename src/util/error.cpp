//
// Created by Stefan on 12/19/2023.
//

#include "util/util.hpp"

namespace util::error {

void ErrNDie(bool cond, const std::string &msg) {
  if (!cond) return;

  std::print("Error: {}\n", msg);
  std::exit(0);
}

}  // namespace util::error
