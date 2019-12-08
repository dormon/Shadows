#pragma once

#include <divRoundUp.h>

template<typename T>
T align(T const&n,T const&d){
  return divRoundUp(n,d) * d;
}
