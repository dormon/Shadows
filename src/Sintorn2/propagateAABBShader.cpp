#include <Sintorn2/propagateAABBShader.h>

std::string const sintorn2::propagateAABBShader = R".(

#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef TILE_X
#define TILE_X 8
#endif//TILE_X

#ifndef TILE_Y
#define TILE_Y 8
#endif//TILE_Y

#ifndef MIN_Z_BITS
#define MIN_Z_BITS 9
#endif//MIN_Z_BITS

layout(local_size_x=WARP)in;

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};
layout(binding=4)buffer ActiveNodes{uint  activeNodes[];};

layout(binding=7)buffer DebugBuffer{uint debugBuffer[];};

uniform uint destLevel = 0;


#if WARP == 64

shared float reductionArray[WARP*6u];

void reduce(){
  float ab[2];
  uint w;

  //6*64 -> 6*32
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  
  ab[0] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[WARP*4u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*4u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];
  
  //6*32 -> 6*16
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&16u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  
  ab[0] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&16u)!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  //6*16 -> 6*8
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&8u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*8 -> 6*4
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&4u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*4 -> 6*2
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&2u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*2 -> 6*1
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&1u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
}

#endif
#line 110
void main(){

#line 141
#if WARP == 64
  uint wid = gl_WorkGroupID.x + gl_WorkGroupID.y*gl_NumWorkGroups.x;

  uint activeNodeUint = activeNodes[nodeLevelOffset[destLevel]+wid];
  uint nodeUint = nodePool[nodeLevelOffsetInUints[destLevel]+activeNodeUint];
  
  //if(gl_LocalInvocationIndex==0){
  //  debugBuffer[wid*2+0] = activeNodeUint;
  //  debugBuffer[wid*2+1] = nodeUint;
  //}

  uint counter = 0;
  while(nodeUint != 0){
    if(counter >=32)break;

    uint bit = findLSB(nodeUint);
    uint node = (activeNodeUint>>1u)*WARP + (activeNodeUint&1u)*halfWarp + bit;
    //if(uint(nodePool[nodeLevelOffsetInUints[destLevel]+(node>>warpBits)*uintsPerWarp + uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u)
    //  return;

    uint isActive = uint(nodePool[nodeLevelOffsetInUints[destLevel+1u]+node*uintsPerWarp + uint(gl_LocalInvocationIndex>31)] & uint(1u<<(uint(gl_LocalInvocationIndex)&0x1fu)));

    uint64_t activeThreads = ballotARB(isActive != 0);
    uint selectedBit       = unpackUint2x32(activeThreads)[0]!=0u?findLSB(unpackUint2x32(activeThreads)[0]):findLSB(unpackUint2x32(activeThreads)[1])+32u;

    if(isActive != 0){
      reductionArray[WARP*0u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+0u];
      reductionArray[WARP*1u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+1u];
      reductionArray[WARP*2u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+2u];
      reductionArray[WARP*3u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+3u];
      reductionArray[WARP*4u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+4u];
      reductionArray[WARP*5u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+5u];
    }

    if(isActive == 0){
      reductionArray[WARP*0u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*0u+selectedBit];
      reductionArray[WARP*1u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*1u+selectedBit];
      reductionArray[WARP*2u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*2u+selectedBit];
      reductionArray[WARP*3u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*3u+selectedBit];
      reductionArray[WARP*4u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*4u+selectedBit];
      reductionArray[WARP*5u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*5u+selectedBit];
    }

    reduce();

    if(gl_LocalInvocationIndex < floatsPerAABB)
      aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];

    nodeUint ^= 1u << bit;
    counter++;
  }

#endif
}

).";
