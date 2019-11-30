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

#ifndef MIN_Z
#define MIN_Z 0
#endif//MIN_Z

#ifndef NEAR
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};
layout(binding=2)buffer AABBCounter{uint  aabbCounter[];};

// converts depth (-1,+1) to Z in view-space
float depthToZ(float d){
  if(isinf(FAR))return 2.f*NEAR    /(d - 1);
  else          return 2.f*NEAR*FAR/(d*(FAR-NEAR)-FAR-NEAR);
}

uint quantizeZ(float z){
  const uint warpBits      = uint(ceil(log2(float(WARP))));
  const uint warpBitsX     = uint(warpBits/2u) + uint(warpBits%2 != 0u);
  const uint warpBitsY     = uint(warpBits-warpBitsX);
  const uint warpX         = uint(1u<<warpBitsX);
  const uint warpY         = uint(1u<<warpBitsY);
  const uint clustersX     = uint(WINDOW_X/warpX) + uint(WINDOW_X%warpX != 0u);
  const uint clustersY     = uint(WINDOW_Y/warpY) + uint(WINDOW_Y%warpY != 0u);
  const uint xBits         = uint(ceil(log2(float(clustersX))));
  const uint yBits         = uint(ceil(log2(float(clustersY))));
  const uint zBits         = MIN_Z>0?MIN_Z:max(max(xBits,yBits),MIN_Z);
  const uint clustersZ     = 1u << zBits;
  const uint Sy            = clustersY;

  return clamp(uint(log(-z/NEAR) / log(1.f+2.f*tan(FOVY/2.f)/Sy)),0,clustersZ-1);
}


uint getMorton(uvec2 coord){
  const uint warpBits      = uint(ceil(log2(float(WARP))));
  const uint warpBitsX     = uint(warpBits/2u) + uint(warpBits%2 != 0u);
  const uint warpBitsY     = uint(warpBits-warpBitsX);

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(warpBitsX,warpBitsY), zQ);
  return morton(clusterCoord);
}

uint activeThread = 0;

shared uint sharedMortons[WARP];

void compute(uvec2 coord){
  const uint warpBits        = uint(ceil(log2(float(WARP))));
  const uint warpBitsX       = uint(warpBits/2u) + uint(warpBits%2 != 0u);
  const uint warpBitsY       = uint(warpBits-warpBitsX);
  const uint warpX           = uint(1u<<warpBitsX);
  const uint warpY           = uint(1u<<warpBitsY);
  const uint clustersX       = uint(WINDOW_X/warpX) + uint(WINDOW_X%warpX != 0u);
  const uint clustersY       = uint(WINDOW_Y/warpY) + uint(WINDOW_Y%warpY != 0u);
  const uint xBits           = uint(ceil(log2(float(clustersX))));
  const uint yBits           = uint(ceil(log2(float(clustersY))));
  const uint zBits           = MIN_Z>0?MIN_Z:max(max(xBits,yBits),MIN_Z);
  const uint allBits         = xBits + yBits + zBits;
  const uint nofLevels       = uint(allBits/warpBits) + uint(allBits%warpBits != 0u);
  const uint uintsPerWarp    = uint(WARP/32);

  const uint warpMask        = uint((1u << warpBits) - 1u);
  const uint uintsPerNode    = uintsPerWarp + 1u + (uintsPerWarp-1);
  const uint lastLevel       = nofLevels - 1u;

  const uint nodePoolLevelOffsets[6] = {
    ((uint(pow(WARP,0))-1)/(WARP-1)) * uintsPerNode,
    ((uint(pow(WARP,1))-1)/(WARP-1)) * uintsPerNode,
    ((uint(pow(WARP,2))-1)/(WARP-1)) * uintsPerNode,
    ((uint(pow(WARP,3))-1)/(WARP-1)) * uintsPerNode,
    ((uint(pow(WARP,4))-1)/(WARP-1)) * uintsPerNode,
    ((uint(pow(WARP,5))-1)/(WARP-1)) * uintsPerNode,
  };

  const uint warpBitsShift [6] = {
    warpBits*0,
    warpBits*1,
    warpBits*2,
    warpBits*3,
    warpBits*4,
    warpBits*5,
  };


  uint morton = getMorton(coord);
  sharedMortons[gl_LocalInvocationIndex] = morton;

  uint notDone;

  if(uintsPerWarp == 1){
    uint notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),0);

    while(notDone != 0){
      uint selectedBit     = findLSB(notDone);
      uint referenceMorton = sharedMortons[selectedBit];

      BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton));
      if(gl_LocalInvocationIndex == 0){
                                nodePool[nodePoolLevelOffsets[nofLevels-1]+(referenceMorton                  )*uintsPerNode] = GET_UINT_FROM_UINT_ARRAY(sameCluster,0);
        if(nofLevels>1)atomicOr(nodePool[nodePoolLevelOffsets[nofLevels-2]+(referenceMorton>>warpBitsShift[1])*uintsPerNode],1u<<((referenceMorton>>warpBitsShift[0])&warpMask));
        if(nofLevels>2)atomicOr(nodePool[nodePoolLevelOffsets[nofLevels-3]+(referenceMorton>>warpBitsShift[2])*uintsPerNode],1u<<((referenceMorton>>warpBitsShift[1])&warpMask));
        if(nofLevels>3)atomicOr(nodePool[nodePoolLevelOffsets[nofLevels-4]+(referenceMorton>>warpBitsShift[3])*uintsPerNode],1u<<((referenceMorton>>warpBitsShift[2])&warpMask));
        if(nofLevels>4)atomicOr(nodePool[nodePoolLevelOffsets[nofLevels-5]+(referenceMorton>>warpBitsShift[4])*uintsPerNode],1u<<((referenceMorton>>warpBitsShift[3])&warpMask));
        if(nofLevels>5)atomicOr(nodePool[nodePoolLevelOffsets[nofLevels-6]+(referenceMorton>>warpBitsShift[5])*uintsPerNode],1u<<((referenceMorton>>warpBitsShift[4])&warpMask));
      }
      notDone ^= sameCluster;
    }
  }
  if(uintsPerWarp == 2){
    DEBUG_SHADER_LINE();
    uint counter = 0;
    for(size_t i=0;i<uintsPerWarp;++i){
      notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),i);
      while(notDone != 0){
        uint selectedBit     = findLSB(notDone) +  i*32u;
        uint referenceMorton = sharedMortons[selectedBit];
        BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton && activeThread != 0));
        if(gl_LocalInvocationIndex == 0){
          for(size_t j=0;j<uintsPerWarp;++j){
            hierarchy["<< offsets.back() << "+referenceMorton*" << uintsPerWarp <<"+"<< j <<"] = GET_UINT_FROM_UINT_ARRAY(sameCluster," << j <<");
          }
          uint bit;
          for(size_t i=0;i<offsets.size()-1;++i){
            auto const offset = offsets[offsets.size()-2-i];
            if(warpBits*(i+1) >= allBits)
              atomicOr(hierarchy[referenceMorton>>5],1<<(referenceMorton&31u));\n";
            else{
              bit = referenceMorton&"<<((1<<warpBits)-1)<<"u;\n";
              referenceMorton >>= " << warpBits << "u;\n";
              atomicOr(hierarchy["<< offset<<"+referenceMorton*"<<uintsPerWarp<<"+(bit>>5)],1<<(bit&31u));\n";
            }
          }
        }
        notDone ^= sameCluster[" << i << "];
      }
    }
  }
}


void main(){
  const uint warpBits      = uint(ceil(log2(float(WARP))));
  const uint warpBitsX     = uint(warpBits/2u) + uint(warpBits%2 != 0u);
  const uint warpBitsY     = uint(warpBits-warpBitsX);
  const uint warpX         = uint(1u<<warpBitsX);
  const uint warpY         = uint(1u<<warpBitsY);
  const uint loCoordMask   = uint((1u<<warpBitsX)-1u);

  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&loCoordMask,uint(gl_LocalInvocationIndex)>>warpBitsX);
  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2(warpX,warpY);
  uvec2 coord = wgCoord + loCoord;
  activeThread = uint(all(lessThan(coord,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord);
}

).";
