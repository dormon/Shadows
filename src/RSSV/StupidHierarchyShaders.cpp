#include <RSSV/StupidHierarchyShaders.h>
#include <GLSLLine.h>


std::string const computeSecondLastLevel =
GLSL_LINE
R".(

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#ifndef NOF_LEVELS
#define NOF_LEVELS 3
#endif//NOF_LEVELS

#define LOCAL_SIZE_IN_FLOATS WAVEFRONT_SIZE

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(binding=0       )uniform sampler2DRect depthTexture;
layout(binding=0,std430)buffer                Level{float level[];};

uniform uint  nofPixels        = 0;
uniform uint  alignedNofPixels = 0;

shared float localDepth[LOCAL_SIZE_IN_FLOATS];

uniform uvec2 nofTiles         = uvec2(0);
uniform uvec2 tileSizeInPixels = uvec2(0);

uniform uvec2 fullTileSizeInPixels[NOF_LEVELS] = {uvec2(512u),uvec2(64u),uvec2( 8u)};
uniform uvec2 fullTileExponent    [NOF_LEVELS] = {uvec2(  9u),uvec2( 6u),uvec2( 3u)};
uniform uvec2 fullTileMask        [NOF_LEVELS] = {uvec2(511u),uvec2(63u),uvec2( 7u)};
uniform uvec2 tileCount           [NOF_LEVELS] = {uvec2(  1u),uvec2( 8u),uvec2(64u)};

float getDepth(uvec2 coord){
  return texelFetch(depthTexture,ivec2(coord),0).r * 2.f - 1.f;
}

float getDepthClamp(uvec2 coord){
  return getDepth(min(windowSize-1,coord));
}

uvec2 threadPixelCoord(){
  uvec2 tileCoord = gl_WorkGroupID.xy * tileSizeInPixels;

  uvec2 localCoord = uvec2(
    gl_LocalInvocationID.x % tileSizeInPixels.x,
    gl_LocalInvocationID.x / tileSizeInPixels.x
    );

  return tileCoord + localCoord;
}

void loadToLocal(){
  uvec2 coord = threadPixelCoord();
  float depth = getDepthClamp(coord);
  localDepth[gl_LocalInvocationID.x] = depth;
}

void findMinMax(){
  uint halfSize = gl_WorkGroupSize.x>>1;
  if(gl_LocalInvocationID.x < halfSize){
    vec2 values;
    values.x = localDepth[gl_LocalInvocationID.x + 0       ];
    values.y = localDepth[gl_LocalInvocationID.x + halfSize];
    uint yLess = uint(values.y<values.x);
    localDepth[gl_LocalInvocationID.x + 0       ] = values[  yLess];
    localDepth[gl_LocalInvocationID.x + halfSize] = values[1-yLess];
  }


  for(;halfSize>0;halfSize>>=1){
    if(gl_LocalInvocationID.x >= halfSize)continue;

    uint quaterSize = halfSize >> 1;
    uint doMax = uint(gl_LocalInvocationID.x >= quaterSize);

    vec2 values;
    uint elem = gl_LocalInvocationID.x & (quaterSize-1);
    values.x = localDepth[doMax*halfSize + elem             ];
    values.y = localDepth[doMax*halfSize + elem + quaterSize];

    uint yLess = uint(values.y<values.x);

    localDepth[elem + doMax*quaterSize] = values[yLess^doMax];

  }
}

#define DIV_ROUND_UP(value,exponent,mask) ((value>>exponent) + uint(greaterThan(value&mask,0)))

uvec2 toSnake(uvec2 coord,uvec2 size,uint yAxis){
  uint odd = coord[yAxis];
  uvec2 result;
  result[1-yAxis] = coord[1-yAxis] + odd*uint(size[1-yAxis] - 2*coord[1-yAxis] - 1u);
  result[  yAxis] = coord[  yAxis]                                                  ;
}

void bridge(){
  if(gl_LocalInvocationID.x > 0)return;

  uvec2 coord;

#define ORIENTATION_BOTTOM 0u
#define ORIENTATION_LEFT   1u
#define ORIENTATION_TOP    2u
#define ORIENTATION_RIGHT  3u
  // >b 00
  // ^l 01
  // <t 10
  // ^r 11
  uint parentOrientation = ORIENTATION_LEFT ;
  uvec2 parentCoord = uvec2(0u);
  uvec2 parentId = uvec2(0u);

#if NOF_LEVELS >= 1
  #if NOF_LEVELS >  1
  parentId = uvec2(gl_WorkGroupID.xy) >> fullTileExponent[0];
  #else
  parentId = uvec2(gl_WorkGroupID.xy);
  #endif
  parentCoord += convertToSnake(parentId,tileCount[0],1u) * tileSizeInPixels[0];
#endif

#if NOF_LEVELS >= 2
  #if NOF_LEVELS >  2
  parentId = uvec2(gl_WorkGroupID.xy) >> fullTileExponent[1];
  #else
  parentId = uvec2(gl_WorkGroupID.xy);
  #endif
  parentCoord += convertToSnake(parentId,tileCount[1],0u) * tileSizeInPixels[1];
#endif

#if NOF_LEVELS >= 3
  #if NOF_LEVELS >  3
  parentId = uvec2(gl_WorkGroupID.xy) >> fullTileExponent[2];
  #else
  parentId = uvec2(gl_WorkGroupID.xy);
  #endif
  parentCoord += convertToSnake(parentId,tileCount[2],1u) * tileSizeInPixels[2];
#endif




  float depth = getDepth(coord);

  localDepth[0] = min(localDepth[0],depth);
  localDepth[1] = min(localDepth[1],depth);
}

void storeToGlobal(){
  if(gl_LocalInvocationID.x > 2)return;

  level[gl_WorkGroupID.x*2+gl_LocalInvocationID.x] = localDepth[gl_LocalInvocationID.x];
}

void main(){
  loadToLocal();

  findMinMax();

  bridge();

  storeToGlobal();  
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
