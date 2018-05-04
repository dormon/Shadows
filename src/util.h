#pragma once

#include<cstdint>

inline uint32_t getDispatchSize(std::size_t n,std::size_t wgs){
  return static_cast<uint32_t>((n/wgs) + ((n%wgs) > 0?1:0));
}

inline std::size_t divRoundUp(std::size_t a,std::size_t b){
  return (a/b) + ((a%b) > 0?1:0);
}

