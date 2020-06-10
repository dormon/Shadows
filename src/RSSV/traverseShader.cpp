#include <RSSV/traverseShader.h>

std::string const rssv::traverseMain = R".(
layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
  #if MEMORY_OPTIM == 1
    uint  aabbPointer[aabbPointerBufferSizeInUints];
  #endif
  #if USE_BRIDGE_POOL == 1
    float bridgePool[bridgePoolSizeInFloats];
  #endif
};

layout(std430,binding=2)buffer JobCounters       {
  uint silhouetteJobCounter;
  uint triangleJobCounter  ;
};

layout(std430,binding=3)readonly buffer EdgePlanes{float edgePlanes       [];};
layout(std430,binding=4)readonly buffer MultBuffer{
  uint nofSilhouettes  ;
  uint multBuffer    [];
};

layout(std430,binding=5)readonly buffer ShadowFrusta{float shadowFrusta[];};

layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

uniform vec4 lightPosition;
uniform vec4 clipLightPosition;

uniform mat4 invTran;
uniform mat4 projView;


#if !defined(SHARED_MEMORY_SIZE) || (SHARED_MEMORY_SIZE) < 0
#undef SHARED_MEMORY_SIZE
#define SHARED_MEMORY_SIZE 0
#endif

#if (SHARED_MEMORY_SIZE) != 0

shared float sharedMemory[(SHARED_MEMORY_SIZE)];

void toShared(in uint offset,in int value){
  sharedMemory[offset] = intBitsToFloat(value);
}

int getShared1i(in uint offset){
  return floatBitsToInt(sharedMemory[offset]);
}

void toShared(in uint offset,in vec4 value){
  sharedMemory[offset+0] = value[0];
  sharedMemory[offset+1] = value[1];
  sharedMemory[offset+2] = value[2];
  sharedMemory[offset+3] = value[3];
}

vec4 getShared4f(in uint offset){
  return vec4(sharedMemory[offset+0],sharedMemory[offset+1],sharedMemory[offset+2],sharedMemory[offset+3]);
}

#endif


#if (STORE_EDGE_PLANES == 1) || (STORE_TRAVERSE_STAT == 1)
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)>>1);
}


#if WARP == 64

shared uint64_t intersection[nofLevels];

#endif


void main(){
  traverseSilhouetteJOB();
}


).";
