#pragma once

#include<sstream>

template<typename T>
std::string toStr(T const&t){
  std::stringstream ss;
  ss << t;
  return ss.str();
}
