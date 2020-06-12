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


// silhouette buffers
#if COMPUTE_SILHOUETTE_PLANES == 1
layout(std430,binding=3)buffer SilhouettePlanes{
  float silhouettePlanes[];
};
#else
layout(std430,binding=3)readonly buffer EdgePlanes{float edgePlanes       [];};
#endif
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
  traverseTriangleJOB();
}


).";
