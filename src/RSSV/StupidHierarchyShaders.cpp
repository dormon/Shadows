#include <RSSV/StupidHierarchyShaders.h>
#include <GLSLLine.h>


std::string const computeSecondLastLevel =
GLSL_LINE
R".(

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#define LOCAL_SIZE_IN_FLOATS (WAVEFRONT_SIZE * 2)

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(binding=0       )uniform sampler2DRect depthTexture;
layout(binding=0,std430)buffer                Level{float level[];};

uniform uint  nofPixels        = 0;
uniform uint  alignedNofPixels = 0;

shared float localDepth[LOCAL_SIZE_IN_FLOATS];

uniform uvec2 nofTiles         = uvec2(0);
uniform uvec2 tileSizeInPixels = uvec2(0);

uvec2 computeTileCoord(){
  uvec2 tileCoord;
  tileCoord.x = gl_WorkGroupID.x % nofTiles.x;
  tileCoord.y = gl_WorkGroupID.x / nofTiles.y;
  if(tileCoord.y&1)tileCoord.y = nofTiles.x - tileCoord.y - 1;
  return tileCoord;
}

uvec2 computeTileCoordInPixels(){
  return computeTileCoord() * tileSizeInPixels;
}

uvec2 computeLocalCoordInPixels(uint evenYTile){
  uvec2 coord;
  coord[evenYTile] = gl_LocalInvocationID[evenYTile] % tileSizeInPixels[evenYTile];
  coord[1-evenYTile] = gl_LocalInvocationID[evenYTile] / tileSizeInPixels[evenYTile];
  if(coord[evenYTile]&1)coord[evenYTile] = tileSizeInPixels[evenYTile] - coord[evenYTile] - 1;
}

void loadToLocal(){
}

void main(){
  if(gl_GlobalInvocationID.x >= nofPixels)
    return;

  loadToLocal();

  ivec2 coord = getCoord(gl_GlobalInvocationID.x);
  float depth = texelFetch(depthTexture,getCoord(gl_GlobalInvocationID.x)).r;

  level[alignedNofPixels*0+gl_GlobalInvocationID.x] = float(coord.x) / float(windowSize.x) * 2 - 1;
  level[alignedNofPixels*1+gl_GlobalInvocationID.x] = float(coord.x) / float(windowSize.x) * 2 - 1;
  level[alignedNofPixels*2+gl_GlobalInvocationID.x] = float(coord.y) / float(windowSize.y) * 2 - 1;
  level[alignedNofPixels*3+gl_GlobalInvocationID.x] = float(coord.y) / float(windowSize.y) * 2 - 1;
  level[alignedNofPixels*4+gl_GlobalInvocationID.x] = depth                                * 2 - 1;
  level[alignedNofPixels*5+gl_GlobalInvocationID.x] = depth                                * 2 - 1;
}

).";

std::string const rssv::copyLevel0Src = 
GLSL_LINE
R".(

#ifndef WAVEFRONT_SIZE
  #define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(binding=0)uniform sampler2DRect depthTexture;
layout(binding=0,std430)buffer Level{float level[];};

uniform uvec2 windowSize;
uniform uint  nofPixels        = 0;
uniform uint  alignedNofPixels = 0;

ivec2 getCoord(uint id){
  const uint line = (id / windowSize.x);
  uint rest = (id%windowSize.x);
  return ivec2(line&1?windowSize.x-rest-1:rest,line);
}

void main(){
  if(gl_GlobalInvocationID.x >= nofPixels)
    return;

  ivec2 coord = getCoord(gl_GlobalInvocationID.x);
  float depth = texelFetch(depthTexture,getCoord(gl_GlobalInvocationID.x)).r;

  level[alignedNofPixels*0+gl_GlobalInvocationID.x] = float(coord.x) / float(windowSize.x) * 2 - 1;
  level[alignedNofPixels*1+gl_GlobalInvocationID.x] = float(coord.x) / float(windowSize.x) * 2 - 1;
  level[alignedNofPixels*2+gl_GlobalInvocationID.x] = float(coord.y) / float(windowSize.y) * 2 - 1;
  level[alignedNofPixels*3+gl_GlobalInvocationID.x] = float(coord.y) / float(windowSize.y) * 2 - 1;
  level[alignedNofPixels*4+gl_GlobalInvocationID.x] = depth                                * 2 - 1;
  level[alignedNofPixels*5+gl_GlobalInvocationID.x] = depth                                * 2 - 1;
}
).";

std::string const rssv::buildNextLevelSrc = 
GLSL_LINE
R".(

#ifndef WAVEFRONT_SIZE
  #define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(binding=0,std430)buffer Input{float inputLevel[];};
layout(binding=1,std430)buffer Output{float outputLevel[];};

uniform uvec2 windowSize;
uniform bool useBridges = false;
uniform uint inputLevelSize = 0;

uniform uint inputNofPixels         = 0;
uniform uint inputAlignedNofPixels  = 0;
uniform uint outputAlignedNofPixels = 0;

ivec2 getCoord(uint id){
  const uint line = (id / windowSize.x);
  uint rest = (id%windowSize.x);
  return ivec2(line&1?windowSize.x-rest:rest-1,line);
}

#define HALF_WAVEFRONT (WAVEFRONT_SIZE/2)

shared float cache[HALF_WAVEFRONT];

float doMinMax(float a,float b,uint doMax){
  /*
  min:
  a<b -> a
  a>b -> b
  a-b < 0 -> a
  a-b > 0 -> b
  max:
  a>b -> a
  a<b -> b
  a-b > 0 -> a
  a-b < 0 -> b
  */
  return (a-b)*(1-2*doMax) < 0?a:b;

}

void main(){
  if(gl_GlobalInvocationID.x >= inputNofPixels)
    return;

  uint wid = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;

  uint offset = wid * WAVEFRONT_SIZE;

  uint partSize;

  for(uint c=0;c<6;++c){
    uint doMax = c&1;
    if(lid < HALF_WAVEFRONT)
      cache[lid] = doMinMax(
        inputLevel[inputAlignedNofPixels*c + wid*WAVEFRONT_SIZE + lid*2+0],
        inputLevel[inputAlignedNofPixels*c + wid*WAVEFRONT_SIZE + lid*2+1],
        doMax);

    for(uint threadsPerLevel = HALF_WAVEFRONT/2;threadsPerLevel>0;threadsPerLevel>>=2)
      for(uint i=0;i<threadsPerLevel;++i)
        cache[i] = doMinMax(cache[i*2+0],cache[i*2+1],doMax);

    if(lid == 0){
      if(useBridges && wid*WAVEFRONT_SIZE + WAVEFRONT_SIZE < partSize)
        outputLevel[outputAlignedNofPixels*c+wid] = doMinMax(inputLevel[inputAlignedNofPixels*c + wid*WAVEFRONT_SIZE + WAVEFRONT_SIZE],cache[0],doMax);
      else
        outputLevel[outputAlignedNofPixels*c+wid] = cache[0];
    }

  }
}

).";
