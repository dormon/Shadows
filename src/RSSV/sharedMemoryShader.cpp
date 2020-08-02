#include <RSSV/sharedMemoryShader.h>

std::string const rssv::sharedMemoryShader = R".(

#if !defined(SHARED_MEMORY_SIZE) || (SHARED_MEMORY_SIZE) < 0
#undef SHARED_MEMORY_SIZE
#define SHARED_MEMORY_SIZE 0
#endif

#if (SHARED_MEMORY_SIZE) != 0

shared float sharedMemory[SHARED_MEMORY_SIZE];

void toShared1i(in uint offset,in int value){
  sharedMemory[offset] = intBitsToFloat(value);
}

int getShared1i(in uint offset){
  return floatBitsToInt(sharedMemory[offset]);
}

uint getShared1u(in uint offset){
  return floatBitsToUint(sharedMemory[offset]);
}

void toShared4f(in uint offset,in vec4 value){
  sharedMemory[offset+0] = value[0];
  sharedMemory[offset+1] = value[1];
  sharedMemory[offset+2] = value[2];
  sharedMemory[offset+3] = value[3];
}

void toShared1f(in uint offset,in float value){
  sharedMemory[offset] = value;
}

void toShared1u(in uint offset,in uint value){
  sharedMemory[offset] = uintBitsToFloat(value);
}

vec4 getShared4f(in uint offset){
  return vec4(sharedMemory[offset+0],sharedMemory[offset+1],sharedMemory[offset+2],sharedMemory[offset+3]);
}

#endif

).";
