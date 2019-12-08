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

#define NOF_WARPS 4
#define THREAD_IN_WARP (gl_LocalInvocationID.x)
#if NOF_WARPS > 1
  #define WARP_ID        (gl_LocalInvocationID.y)
#else
  #define WARP_ID        0
#endif
#define WARP_OFFSET    (WARP_ID*6u*WARP)

layout(local_size_x=WARP,local_size_y=NOF_WARPS)in;

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};
layout(binding=4)buffer ActiveNodes{uint  activeNodes[];};

layout(binding=7)buffer DebugBuffer{uint debugBuffer[];};

uniform uint destLevel = 0;


#if WARP == 64

shared float reductionArray[WARP*6u*NOF_WARPS];



void reduce(){
  float ab[2];
  uint w;

  //6*64 -> 6*32
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&32u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
  
  ab[0] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&32u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*1u + THREAD_IN_WARP] = ab[w];

  ab[0] = reductionArray[WARP_OFFSET+WARP*4u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*4u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&32u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*2u + THREAD_IN_WARP] = ab[w];
  
  //6*32 -> 6*16
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&16u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
  
  ab[0] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&16u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*1u + THREAD_IN_WARP] = ab[w];

  //6*16 -> 6*8
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&8u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];

  //6*8 -> 6*4
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&4u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];

  //6*4 -> 6*2
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&2u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];

  //6*2 -> 6*1
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&1u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
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

#if NOF_WARPS > 1
  nodeUint &= uint((1u<<uint(32/NOF_WARPS))-1u) << (uint(32/NOF_WARPS)*WARP_ID);
#endif


#ifdef BE_SAFE
  uint counter = 0;
#endif

  while(nodeUint != 0){

#ifdef BE_SAFE
    if(counter >=32)break;
#endif

    uint bit = findLSB(nodeUint);
    uint node = (activeNodeUint>>1u)*WARP + (activeNodeUint&1u)*halfWarp + bit;

    uint isActive = uint(nodePool[nodeLevelOffsetInUints[destLevel+1u]+node*uintsPerWarp + uint(THREAD_IN_WARP>31)] & uint(1u<<(uint(THREAD_IN_WARP)&0x1fu)));

    uint64_t activeThreads = ballotARB(isActive != 0);
    uint selectedBit       = unpackUint2x32(activeThreads)[0]!=0u?findLSB(unpackUint2x32(activeThreads)[0]):findLSB(unpackUint2x32(activeThreads)[1])+32u;

    if(isActive != 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+0u];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+1u];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+2u];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+3u];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+4u];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+5u];
    }

    if(isActive == 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*0u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*1u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*2u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*3u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*4u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*5u+selectedBit];
    }

    reduce();

    if(THREAD_IN_WARP < floatsPerAABB)
      aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+THREAD_IN_WARP] = reductionArray[WARP_OFFSET+THREAD_IN_WARP];

    nodeUint ^= 1u << bit;

#ifdef BE_SAFE
    counter++;
#endif
  }

#endif
}

).";
