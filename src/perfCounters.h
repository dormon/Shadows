#pragma once

#include <functional>

namespace perf{
  void printComputeShaderProf(std::function<void()>const&fce);
  void printComputeShaderProf(std::function<void()>const&fce,uint32_t counter);
}
