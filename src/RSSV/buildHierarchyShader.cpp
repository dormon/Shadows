#include <RSSV/buildHierarchyShader.h>

std::string const rssv::buildHierarchyShader = R".(
#line 237
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

#ifndef NEAR
#define NEAR 1.f
#endif//NEAR

#ifndef FAR
#define FAR 100.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

#ifndef USE_PADDING
#define USE_PADDING 0
#endif//USE_PADDING

layout(local_size_x=WARP)in;

#if MERGED_BUFFERS == 1
layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
};
#else
layout(std430,binding=0)buffer NodePool        {uint  nodePool        [];};
layout(std430,binding=1)buffer AABBPool        {float aabbPool        [];};
#endif


layout(std430,binding=3)buffer LevelNodeCounter{uint  levelNodeCounter[];};
layout(std430,binding=4)buffer ActiveNodes     {uint  activeNodes     [];};

#if MEMORY_OPTIM == 1
layout(std430,binding=5)buffer AABBPointer     {uint  aabbPointer     [];};
#endif

layout(binding=1)uniform sampler2DRect depthTexture;

#if DISCARD_BACK_FACING == 1
layout(binding=2)uniform sampler2D     normalTexture;
uniform vec4 lightPosition;
#endif

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = DEPTH_TO_Z(depth);
  uint  zQ = QUANTIZE_Z(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}



#if WARP == 64
uint activeThread = 0;
#else
uint activeThread[2] = {0,0};
#endif


#if WARP == 64
void compute(uvec2 coord){
#else
void compute(uvec2 coord,uvec2 coord2){
#endif




#if WARP==64
  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);
  activeThread &= uint(depth < 1.f);

#if DISCARD_BACK_FACING == 1
  activeThread &= uint(dot(lightPosition,texelFetch(normalTexture,ivec2(coord),0))>0);
#endif

#else
  float depth [2];
  uint  morton[2];
  depth [0] = texelFetch(depthTexture,ivec2(coord )).x*2-1;
  depth [1] = texelFetch(depthTexture,ivec2(coord2)).x*2-1;
  morton[0] = getMorton(coord ,depth[0]);
  morton[1] = getMorton(coord2,depth[1]);
#endif

#line 322
  //if(uintsPerWarp == 1){
  #if WARP == 32
    uint counter = 0;
    uint notDone[2];
    notDone[0] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread[0] != 0)),0);
    notDone[1] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread[1] != 0)),0);
    while(notDone[0] != 0 || notDone[1] != 0){

      if(counter >= (TILE_X*TILE_Y))break;
      counter ++;

      uint selectedBit     = notDone[0]!=0?findLSB(notDone[0]):findLSB(notDone[1])+32u;
      uint referenceMorton = readInvocationARB(morton[uint(selectedBit>31u)],selectedBit&uint(0x1fu));

      if(gl_LocalInvocationIndex == 0){
        if(nofLevels>0){
          uint bit  = (referenceMorton >> (warpBits*0u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*1u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>1){
          uint bit  = (referenceMorton >> (warpBits*1u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*2u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
          if(mm == 0){
            mm = atomicAdd(levelNodeCounter[clamp(nofLevels-2u,0u,5u)*4u],1);
            activeNodes[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+mm] = node*uintsPerWarp;
          }
        }
        //if(nofLevels>2){
        //  uint bit  = (referenceMorton >> (warpBits*2u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*3u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-3u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>3){
        //  uint bit  = (referenceMorton >> (warpBits*3u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*4u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-4u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>4){
        //  uint bit  = (referenceMorton >> (warpBits*4u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*5u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-5u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>5){
        //  uint bit  = (referenceMorton >> (warpBits*5u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*6u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-6u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
      }

      uint sameCluster[2];
      sameCluster[0] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton[0])),0);
      sameCluster[1] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton[1])),0);

      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+0   ] = -1.f + 2.f/float(WINDOW_X)*(coord.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+0   ] = -1.f + 2.f/float(WINDOW_Y)*(coord.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+0   ] = depth[0];
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+WARP] = -1.f + 2.f/float(WINDOW_X)*(coord2.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+WARP] = -1.f + 2.f/float(WINDOW_Y)*(coord2.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+WARP] = depth[1];

      memoryBarrierShared();

      if(referenceMorton != morton[0] || activeThread[0] == 0){
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*0u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*1u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*2u];
      }

      if(referenceMorton != morton[1] || activeThread[1] == 0){
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*0u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*1u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*2u];
      }
      memoryBarrierShared();

      reduce();

#if MEMORY_OPTIM == 1
      if(gl_LocalInvocationIndex==0){
        uint w = atomicAdd(aabbPointer[0],1);
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPointer[nodeLevelOffset[clamp(nofLevels-1u,0u,5u)]+node+1] = w;
        aabbPool[w*6+0] = reductionArray[0];
        aabbPool[w*6+1] = reductionArray[1];
        aabbPool[w*6+2] = reductionArray[2];
        aabbPool[w*6+3] = reductionArray[3];
        aabbPool[w*6+4] = reductionArray[4];
        aabbPool[w*6+5] = reductionArray[5];
      }
#else
      if(gl_LocalInvocationIndex < floatsPerAABB){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];
      }
#endif

      notDone[0] ^= sameCluster[0];
      notDone[1] ^= sameCluster[1];
    }
  #endif
  //}
 

#line 166
  #if WARP == 64
    uint counter = 0;
    uint64_t notDone = ballotARB(activeThread != 0);
    while(notDone != 0){

      if(counter >= WARP)break;

      uint selectedBit     = unpackUint2x32(notDone)[0]!=0?findLSB(unpackUint2x32(notDone)[0]):findLSB(unpackUint2x32(notDone)[1])+32u;
      uint referenceMorton = readInvocationARB(morton,selectedBit);

      if(gl_LocalInvocationIndex == 0){

        if(nofLevels>0){
          uint bit  = (referenceMorton >> (warpBits*0u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*1u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>1){
          uint bit  = (referenceMorton >> (warpBits*1u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*2u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
          if(mm == 0){
            mm = atomicAdd(levelNodeCounter[clamp(nofLevels-2u,0u,5u)*4u],1);
            activeNodes[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+mm] = node*uintsPerWarp+uint(bit>31u);
          }
        }

      }
      
      uint64_t sameCluster = ballotARB(referenceMorton == morton && activeThread != 0);

      reductionArray[gl_LocalInvocationIndex+WARP*0u] = -1.f + 2.f/float(WINDOW_X)*(coord.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+WARP*1u] = -1.f + 2.f/float(WINDOW_Y)*(coord.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+WARP*2u] = depth;

      if(referenceMorton != morton || activeThread == 0){
        reductionArray[gl_LocalInvocationIndex+WARP*0u] = reductionArray[selectedBit+WARP*0u];
        reductionArray[gl_LocalInvocationIndex+WARP*1u] = reductionArray[selectedBit+WARP*1u];
        reductionArray[gl_LocalInvocationIndex+WARP*2u] = reductionArray[selectedBit+WARP*2u];
      }

      reduce();


#if MEMORY_OPTIM == 1
      if(gl_LocalInvocationIndex==0){
        uint w = atomicAdd(aabbPointer[0],1);
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPointer[nodeLevelOffset[clamp(nofLevels-1u,0u,5u)]+node+1] = w;
#if USE_PADDING == 1
        float aaa = (CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5]))+1) - CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])))) / 32.f + CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])));
        float bbb = CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])));
        float ppp = Z_TO_DEPTH(aaa) - Z_TO_DEPTH(bbb);

        const float size[6] = {
          -0.5f/float(WINDOW_X),
          +0.5f/float(WINDOW_X),
          -0.5f/float(WINDOW_Y),
          +0.5f/float(WINDOW_Y),
          -ppp,
          +ppp,
        };

        aabbPool[w*6+0] = reductionArray[0] + size[0];
        aabbPool[w*6+1] = reductionArray[1] + size[1];
        aabbPool[w*6+2] = reductionArray[2] + size[2];
        aabbPool[w*6+3] = reductionArray[3] + size[3];
        aabbPool[w*6+4] = reductionArray[4] + size[4];
        aabbPool[w*6+5] = reductionArray[5] + size[5];
#else
        aabbPool[w*6+0] = reductionArray[0];
        aabbPool[w*6+1] = reductionArray[1];
        aabbPool[w*6+2] = reductionArray[2];
        aabbPool[w*6+3] = reductionArray[3];
        aabbPool[w*6+4] = reductionArray[4];
        aabbPool[w*6+5] = reductionArray[5];
#endif
      }
#else
      if(gl_LocalInvocationIndex < floatsPerAABB){
        uint node = (referenceMorton >> (warpBits*0u));
#if USE_PADDING == 1

        float aaa = (CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5]))+1) - CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])))) / 32.f + CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])));
        float bbb = CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(reductionArray[5])));
        float ppp = Z_TO_DEPTH(aaa) - Z_TO_DEPTH(bbb);

        const float size[6] = {
          -0.5f/float(WINDOW_X),
          +0.5f/float(WINDOW_X),
          -0.5f/float(WINDOW_Y),
          +0.5f/float(WINDOW_Y),
          -ppp,
          +ppp,
        };
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex] + size[gl_LocalInvocationIndex];
#else
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];
#endif
      }

#endif



      notDone ^= sameCluster;
      counter++;
    }

  #endif
}


void main(){
  const uint loCoordShift  = uint(ceil(log2(float(TILE_X))));
  const uint loCoordMask   = uint(TILE_X-1u);

  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&loCoordMask,uint(gl_LocalInvocationIndex)>>loCoordShift);
  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2(TILE_X,TILE_Y);

#if WARP == 64
  uvec2 coord = wgCoord + loCoord;
  activeThread = uint(all(lessThan(coord,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord);
#else
  uvec2 coord  = wgCoord + loCoord;
  uvec2 coord2 = wgCoord + loCoord + uvec2(0,4u);
  activeThread[0] = uint(all(lessThan(coord ,uvec2(WINDOW_X,WINDOW_Y))));
  activeThread[1] = uint(all(lessThan(coord2,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord,coord2);
#endif
}

).";
