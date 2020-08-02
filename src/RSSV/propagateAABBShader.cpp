#include <RSSV/propagateAABBShader.h>

std::string const rssv::propagateAABBShader = R".(

#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef NOF_WARPS
#define NOF_WARPS 4
#endif//NOF_WARPS


#define THREAD_IN_WARP (gl_LocalInvocationID.x)
#if NOF_WARPS > 1
  #define WARP_ID        (gl_LocalInvocationID.y)
#else
  #define WARP_ID        0
#endif
#define WARP_OFFSET    (WARP_ID*6u*WARP)

layout(local_size_x=WARP,local_size_y=NOF_WARPS)in;

layout(std430,binding=0)buffer NodePool{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
  #if MEMORY_OPTIM == 1
  uint  aabbPointer[aabbPointerBufferSizeInUints];
  #endif
  #if USE_BRIDGE_POOL == 1
  float  bridgePool[bridgePoolSizeInFloats];
  #endif
};

layout(std430,binding=3)buffer LevelNodeCounter{uint  levelNodeCounter[];};
layout(std430,binding=4)buffer ActiveNodes     {uint  activeNodes     [];};
layout(std430,binding=6)buffer Bridges           { int  bridges          [];};


layout(std430,binding=7)buffer DebugBuffer{uint debugBuffer[];};

uniform uint destLevel = 0;

#if WARP == 32

shared float reductionArray[WARP*6u*NOF_WARPS];

void reduce(){
  float ab[2];
  uint w;

  //6*32 -> 6*16
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&16u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
  
  ab[0] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&16u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*1u + THREAD_IN_WARP] = ab[w];

  ab[0] = reductionArray[WARP_OFFSET+WARP*4u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*4u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&16u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*2u + THREAD_IN_WARP] = ab[w];
memoryBarrierShared();
  
  //6*16 -> 6*8
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&8u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
  
  ab[0] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*2u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&8u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*1u + THREAD_IN_WARP] = ab[w];
memoryBarrierShared();

  //6*8 -> 6*4
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&4u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
memoryBarrierShared();

  //6*4 -> 6*2
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&2u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
memoryBarrierShared();

  //6*2 -> 6*1
  ab[0] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP_OFFSET+WARP*0u + (uint(THREAD_IN_WARP)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(THREAD_IN_WARP)&1u)!=0u)) > 0.f);
  reductionArray[WARP_OFFSET+WARP*0u + THREAD_IN_WARP] = ab[w];
memoryBarrierShared();

}

#endif

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

#if WARP == 32

  uint wid = gl_WorkGroupID.x + gl_WorkGroupID.y*gl_NumWorkGroups.x;

  uint activeNodeUint = activeNodes[nodeLevelOffset[destLevel]+wid];
  uint nodeUint = nodePool[nodeLevelOffsetInUints[destLevel]+activeNodeUint];

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
    uint node = activeNodeUint*WARP + bit;

    uint isActive = uint(nodePool[nodeLevelOffsetInUints[destLevel+1u]+node*uintsPerWarp + uint(THREAD_IN_WARP>31)] & uint(1u<<(uint(THREAD_IN_WARP)&0x1fu)));

    uint64_t activeThreads = ballotARB(isActive != 0);
    uint selectedBit       = findLSB(unpackUint2x32(activeThreads)[0]);

#if MEMORY_OPTIM == 1
    if(isActive != 0){
      uint w = aabbPointer[nodeLevelOffset[destLevel+1]+node*WARP+uint(THREAD_IN_WARP)+1u];
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+0u];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+1u];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+2u];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+3u];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+4u];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+5u];
    }
#else
    if(isActive != 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+0u];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+1u];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+2u];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+3u];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+4u];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+5u];
    }
#endif

    memoryBarrierShared();

    if(isActive == 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*0u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*1u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*2u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*3u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*4u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*5u+selectedBit];
    }
    memoryBarrierShared();

    reduce();

#if MEMORY_OPTIM == 1
    if(THREAD_IN_WARP==0){
      uint w = atomicAdd(aabbPointer[0],1);
      aabbPointer[nodeLevelOffset[destLevel]+node+1] = w;
      aabbPool[w*6+0] = reductionArray[WARP_OFFSET+0];
      aabbPool[w*6+1] = reductionArray[WARP_OFFSET+1];
      aabbPool[w*6+2] = reductionArray[WARP_OFFSET+2];
      aabbPool[w*6+3] = reductionArray[WARP_OFFSET+3];
      aabbPool[w*6+4] = reductionArray[WARP_OFFSET+4];
      aabbPool[w*6+5] = reductionArray[WARP_OFFSET+5];
    }
#else
    if(THREAD_IN_WARP < floatsPerAABB)
      aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+THREAD_IN_WARP] = reductionArray[WARP_OFFSET+THREAD_IN_WARP];
#endif
    if(THREAD_IN_WARP == 0)
      bridges[nodeLevelOffset[destLevel] + node] = 0;

    nodeUint ^= 1u << bit;

#ifdef BE_SAFE
    counter++;
#endif
  }

  if(gl_LocalInvocationIndex == 0){

    uint bit  = activeNodeUint& warpMask;
    uint node = activeNodeUint>>warpBits;

    if(destLevel > 0){
      uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[destLevel-1]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
      if(mm == 0){
        mm = atomicAdd(levelNodeCounter[(destLevel-1)*4u],1);
        activeNodes[nodeLevelOffset[destLevel-1]+mm] = node*uintsPerWarp+uint(bit>31u);
      }
    }
  }


#endif


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

#if MEMORY_OPTIM == 1
    if(isActive != 0){
      uint w = aabbPointer[nodeLevelOffset[destLevel+1]+node*WARP+uint(THREAD_IN_WARP)+1u];
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+0u];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+1u];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+2u];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+3u];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+4u];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = aabbPool[w*floatsPerAABB+5u];
    }
#else
    if(isActive != 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+0u];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+1u];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+2u];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+3u];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+4u];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(THREAD_IN_WARP)*floatsPerAABB+5u];
    }
#endif

    if(isActive == 0){
      reductionArray[WARP_OFFSET+WARP*0u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*0u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*1u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*1u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*2u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*2u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*3u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*3u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*4u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*4u+selectedBit];
      reductionArray[WARP_OFFSET+WARP*5u+uint(THREAD_IN_WARP)] = reductionArray[WARP_OFFSET+WARP*5u+selectedBit];
    }

    reduce();

#if MEMORY_OPTIM == 1
    if(THREAD_IN_WARP==0){
      uint w = atomicAdd(aabbPointer[0],1);
      aabbPointer[nodeLevelOffset[destLevel]+node+1] = w;
      aabbPool[w*6+0] = reductionArray[WARP_OFFSET+0];
      aabbPool[w*6+1] = reductionArray[WARP_OFFSET+1];
      aabbPool[w*6+2] = reductionArray[WARP_OFFSET+2];
      aabbPool[w*6+3] = reductionArray[WARP_OFFSET+3];
      aabbPool[w*6+4] = reductionArray[WARP_OFFSET+4];
      aabbPool[w*6+5] = reductionArray[WARP_OFFSET+5];
#if USE_BRIDGE_POOL == 1
      bridgePool[w*floatsPerBridge+0] = (reductionArray[WARP_OFFSET+0] + reductionArray[WARP_OFFSET+1])*.5f;
      bridgePool[w*floatsPerBridge+1] = (reductionArray[WARP_OFFSET+2] + reductionArray[WARP_OFFSET+3])*.5f;
      bridgePool[w*floatsPerBridge+2] = (reductionArray[WARP_OFFSET+4] + reductionArray[WARP_OFFSET+5])*.5f;
#endif
    }
#else
    if(THREAD_IN_WARP < floatsPerAABB)
      aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+THREAD_IN_WARP] = reductionArray[WARP_OFFSET+THREAD_IN_WARP];

#if USE_BRIDGE_POOL == 1
    if(THREAD_IN_WARP < floatsPerBridge)
      bridgePool[bridgeLevelOffsetInFloats[destLevel]+node*floatsPerBridge+THREAD_IN_WARP] = (reductionArray[WARP_OFFSET+THREAD_IN_WARP*2+0] + reductionArray[WARP_OFFSET+THREAD_IN_WARP*2+1])*0.5f;
#endif

#endif
    if(THREAD_IN_WARP == 0)
      bridges[nodeLevelOffset[destLevel] + node] = 0;


    nodeUint ^= 1u << bit;

#ifdef BE_SAFE
    counter++;
#endif
  }

  if(gl_LocalInvocationIndex == 0){

    uint bit  = (activeNodeUint>>1u)& warpMask;
    uint node = (activeNodeUint>>1u)>>warpBits;


    if(destLevel > 0){
      uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[destLevel-1]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
      if(mm == 0){
        mm = atomicAdd(levelNodeCounter[(destLevel-1)*4u],1);
        activeNodes[nodeLevelOffset[destLevel-1]+mm] = node*uintsPerWarp+uint(bit>31u);
      }
    }
  }

#endif
}

).";
