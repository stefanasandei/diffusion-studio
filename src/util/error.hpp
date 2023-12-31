//
// Created by Stefan on 12/19/2023.
//

#pragma once

#include "util/std.hpp"

namespace util::error {

void ErrNDie(bool cond, const std::string &msg);

}