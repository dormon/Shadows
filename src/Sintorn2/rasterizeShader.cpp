#include <Sintorn2/rasterizeShader.h>

std::string const sintorn2::rasterizeShader = R".(

#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef NOF_TRIANGLES
#define NOF_TRIANGLES 0u
#endif//NOF_TRIANGLES

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#define PLANES_PER_SF 4
#define FLOATS_PER_PLANE 4
#define FLOATS_PER_SF (PLANES_PER_SF * FLOATS_PER_PLANE)

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool    {uint  nodePool    [];};
layout(std430,binding=1)buffer AABBPool    {float aabbPool    [];};
layout(std430,binding=2)buffer ShadowFrusta{float shadowFrusta[];};
layout(std430,binding=3)buffer JobCounter  {uint  jobCounter  [];};


const uint alignedNofSF = (uint(NOF_TRIANGLES / SF_ALIGNMENT) + uint((NOF_TRIANGLES % SF_ALIGNMENT) != 0u)) * SF_ALIGNMENT;

float shadowFrustaPlanes[4*4];
#line 36
void main(){
  uint job;
  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)return;

    if(gl_LocalInvocationIndex < FLOATS_PER_SF){
#if TRIANGLE_INTERLEAVE == 1
      shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[alignedNofSF*gl_LocalInvocationIndex + job];
#else
      shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[job*FLOATS_PER_SF+gl_LocalInvocationIndex];
#endif
    }

  }
}

).";
