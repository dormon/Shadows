#include <RSSV/mergeShader.h>

std::string const rssv::mergeShaderFWD = R".(
void mergeJOB();
).";

std::string const rssv::mergeShader = R".(

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = DEPTH_TO_Z(depth);
  uint  zQ = QUANTIZE_Z(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}

void merge(uint job){
  uvec2 tile       = uvec2(job%clustersX,job/clustersX);
  uvec2 localCoord = uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);
  uvec2 coord      = tile*uvec2(TILE_X,TILE_Y) + localCoord;
  if(any(greaterThanEqual(coord,uvec2(WINDOW_X,WINDOW_Y))))return;
  int mult = imageLoad(stencil,ivec2(coord)).x;

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);

  if(nofLevels>0){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-1u,0u,5u)]+(morton>>(warpBits*0))];
  }
  if(nofLevels>1){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+(morton>>(warpBits*1))];
  }
  if(nofLevels>2){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-3u,0u,5u)]+(morton>>(warpBits*2))];
  }
  if(nofLevels>3){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-4u,0u,5u)]+(morton>>(warpBits*3))];
  }
  if(nofLevels>4){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-5u,0u,5u)]+(morton>>(warpBits*4))];
  }
  if(nofLevels>5){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-6u,0u,5u)]+(morton>>(warpBits*5))];
  }

  if(mult != 0)
    imageStore(shadowMask,ivec2(coord),vec4(0.f));
}


void mergeJOB(){
#if PERFORM_MERGE == 1
  uint job = 0u;
  for(;;){
  
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(mergeJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= clustersX*clustersY)break;

    merge(job);
  }
#endif
}
).";

