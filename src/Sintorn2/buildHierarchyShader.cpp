#include <Sintorn2/buildHierarchyShader.h>

std::string const sintorn2::buildHierarchyShader = R".(
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
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

layout(local_size_x=WARP)in;

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};
layout(binding=2)buffer AABBCounter{uint  aabbCounter[];};
layout(binding=1)uniform sampler2DRect depthTexture;


// converts depth (-1,+1) to Z in view-space
float depthToZ(float d){
#ifdef FAR_IS_INFINITE
  return 2.f*NEAR    /(d - 1);
#else
  return 2.f*NEAR*FAR/(d*(FAR-NEAR)-FAR-NEAR);
#endif
}

uint quantizeZ(float z){
  const uint clustersX     = uint(WINDOW_X/TILE_X) + uint(WINDOW_X%TILE_X != 0u);
  const uint clustersY     = uint(WINDOW_Y/TILE_Y) + uint(WINDOW_Y%TILE_Y != 0u);
  const uint xBits         = uint(ceil(log2(float(clustersX))));
  const uint yBits         = uint(ceil(log2(float(clustersY))));
  const uint zBits         = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
  const uint clustersZ     = 1u << zBits;
  const uint Sy            = clustersY;

  return clamp(uint(log(-z/NEAR) / log(1.f+2.f*tan(FOVY/2.f)/Sy)),0,clustersZ-1);
}


uint getMorton(uvec2 coord){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}

uint activeThread = 0;

shared uint sharedMortons[WARP];


void compute(uvec2 coord){
  const uint warpBits        = uint(ceil(log2(float(WARP))));
  const uint clustersX       = uint(WINDOW_X/TILE_X) + uint(WINDOW_X%TILE_X != 0u);
  const uint clustersY       = uint(WINDOW_Y/TILE_Y) + uint(WINDOW_Y%TILE_Y != 0u);
  const uint xBits           = uint(ceil(log2(float(clustersX))));
  const uint yBits           = uint(ceil(log2(float(clustersY))));
  const uint zBits           = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
  const uint allBits         = xBits + yBits + zBits;
  const uint nofLevels       = uint(allBits/warpBits) + uint(allBits%warpBits != 0u);
  const uint uintsPerWarp    = uint(WARP/32);

  const uint warpMask        = uint((1u << warpBits) - 1u);
  const uint uintsPerNode    = uintsPerWarp + 1u + (uintsPerWarp-1);

  const uint levelSize[6] = {
    1u << (allBits >> ((nofLevels-0u)*warpBits)),
    1u << (allBits >> ((nofLevels-1u)*warpBits)),
    1u << (allBits >> ((nofLevels-2u)*warpBits)),
    1u << (allBits >> ((nofLevels-3u)*warpBits)),
    1u << (allBits >> ((nofLevels-4u)*warpBits)),
    1u << (allBits >> ((nofLevels-5u)*warpBits)),
  };

  const uint levelOffset[6] = {
    0,
    0 + levelSize[0],
    0 + levelSize[0] + levelSize[1],
    0 + levelSize[0] + levelSize[1] + levelSize[2],
    0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3],
    0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3] + levelSize[4],
  };

  /*
  if(gl_GlobalInvocationID.x == 0){
    uint sx=300;
    uint sy=300;
    for(uint x=sx;x<512;++x)
      for(uint y=sx;y<512;++y){

        float depth = texelFetch(depthTexture,ivec2(x,y)).x*2.f-1.f;
        float z = depthToZ(depth);
        //nodePool[(y-sy)*(512-sx)+(x-sx)] = floatBitsToUint(z);
        uint  zQ = quantizeZ(z);
        nodePool[(y-sy)*(512-sx)+(x-sx)] = zQ;
      }
  }
  return;
  */

  uint morton = getMorton(coord);
  sharedMortons[gl_LocalInvocationIndex] = morton;
#line 120
  //if(uintsPerWarp == 1){
  #if WARP == 32
    uint notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),0);
  
    uint counter = 0;
    while(notDone != 0){
      if(counter >= 32)break;
      counter ++;

      uint selectedBit     = findLSB(notDone);
      uint referenceMorton = sharedMortons[selectedBit];

      uint sameCluster = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton)),0);
      if(gl_LocalInvocationIndex == 0){

        //if(nofLevels>0)atomicOr(nodePool[(referenceMorton>>(warpBits*1))],1u<<((referenceMorton>>(warpBits*0))&warpMask));
        if(nofLevels>0)atomicOr(nodePool[levelOffset[clamp(nofLevels-1u,0u,5u)]+(referenceMorton>>(warpBits*1))],1u<<((referenceMorton>>(warpBits*0))&warpMask));
        if(nofLevels>1)atomicOr(nodePool[levelOffset[clamp(nofLevels-2u,0u,5u)]+(referenceMorton>>(warpBits*2))],1u<<((referenceMorton>>(warpBits*1))&warpMask));
        if(nofLevels>2)atomicOr(nodePool[levelOffset[clamp(nofLevels-3u,0u,5u)]+(referenceMorton>>(warpBits*3))],1u<<((referenceMorton>>(warpBits*2))&warpMask));
        if(nofLevels>3)atomicOr(nodePool[levelOffset[clamp(nofLevels-4u,0u,5u)]+(referenceMorton>>(warpBits*4))],1u<<((referenceMorton>>(warpBits*3))&warpMask));
        if(nofLevels>4)atomicOr(nodePool[levelOffset[clamp(nofLevels-5u,0u,5u)]+(referenceMorton>>(warpBits*5))],1u<<((referenceMorton>>(warpBits*4))&warpMask));
        if(nofLevels>5)atomicOr(nodePool[levelOffset[clamp(nofLevels-6u,0u,5u)]+(referenceMorton>>(warpBits*6))],1u<<((referenceMorton>>(warpBits*5))&warpMask));
      }
      notDone ^= sameCluster;
    }
  #endif
  //}
 
  //if(uintsPerWarp == 2){
  #if WARP == 64
    uint counter = 0;
    BALLOT_UINTS notDone = BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0));
    while(notDone[0] != 0 || notDone[1] != 0){
      uint selectedBit     = notDone[0]!=0?findLSB(notDone[0]):findLSB(notDone[1])+32;
      uint referenceMorton = sharedMortons[selectedBit];

      BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton && activeThread != 0));
      if(gl_LocalInvocationIndex == 0){
                                nodePool[nodePoolLevelOffsets[nofLevels-1u]+(referenceMorton                  )*uintsPerNode+0] = sameCluster[0];
                                nodePool[nodePoolLevelOffsets[nofLevels-1u]+(referenceMorton                  )*uintsPerNode+1] = sameCluster[1];
        if(nofLevels>1){
          uint shift = (referenceMorton>>warpBitsShift[0])&warpMask;
          atomicOr(nodePool[nodePoolLevelOffsets[clamp(nofLevels-2u,0u,5u)]+(referenceMorton>>warpBitsShift[1])*uintsPerNode+uint(shift>31u)],1u<<(shift&0x1fu));
        }
        if(nofLevels>2){
          uint shift = (referenceMorton>>warpBitsShift[1])&warpMask;
          atomicOr(nodePool[nodePoolLevelOffsets[clamp(nofLevels-3u,0u,5u)]+(referenceMorton>>warpBitsShift[2])*uintsPerNode+uint(shift>31u)],1u<<(shift&0x1fu));
        }
        if(nofLevels>3){
          uint shift = (referenceMorton>>warpBitsShift[2])&warpMask;
          atomicOr(nodePool[nodePoolLevelOffsets[clamp(nofLevels-4u,0u,5u)]+(referenceMorton>>warpBitsShift[3])*uintsPerNode+uint(shift>31u)],1u<<(shift&0x1fu));
        }
        if(nofLevels>4){
          uint shift = (referenceMorton>>warpBitsShift[3])&warpMask;
          atomicOr(nodePool[nodePoolLevelOffsets[clamp(nofLevels-5u,0u,5u)]+(referenceMorton>>warpBitsShift[4])*uintsPerNode+uint(shift>31u)],1u<<(shift&0x1fu));
        }
        if(nofLevels>5){
          uint shift = (referenceMorton>>warpBitsShift[4])&warpMask;
          atomicOr(nodePool[nodePoolLevelOffsets[clamp(nofLevels-6u,0u,5u)]+(referenceMorton>>warpBitsShift[5])*uintsPerNode+uint(shift>31u)],1u<<(shift&0x1fu));
        }
      }
      notDone[0] ^= sameCluster[0];
      notDone[1] ^= sameCluster[1];
    }
  #endif
  //}
}


void main(){
  const uint loCoordShift  = uint(ceil(log2(float(TILE_X))));
  const uint loCoordMask   = uint(TILE_X-1u);

  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&loCoordMask,uint(gl_LocalInvocationIndex)>>loCoordShift);
  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2(TILE_X,TILE_Y);
  uvec2 coord = wgCoord + loCoord;
  activeThread = uint(all(lessThan(coord,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord);
}

).";
