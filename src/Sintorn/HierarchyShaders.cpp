#include <Sintorn/HierarchyShaders.h>
#include <GLSLLine.h>

const std::string sintorn::writeDepthSrc = 
GLSL_LINE
R".(
//methods/Sintorn/writedepthtexture.comp";
//DO NOT EDIT ANYTHING BELOW THIS LINE

#ifndef WRITEDEPTHTEXTURE_BINDING_DEPTH
  #define WRITEDEPTHTEXTURE_BINDING_DEPTH 0
#endif//WRITEDEPTHTEXTURE_BINDING_DEPTH

#ifndef WRITEDEPTHTEXTURE_BINDING_HDT
  #define WRITEDEPTHTEXTURE_BINDING_HDT   1
#endif//WRITEDEPTHTEXTURE_BINDING_HDT

#ifndef WRITEDEPTHTEXTURE_BINDING_NORMAL
  #define WRITEDEPTHTEXTURE_BINDING_NORMAL   2
#endif//WRITEDEPTHTEXTURE_BINDING_NORMAL

#ifndef LOCAL_TILE_SIZE_X
  #define LOCAL_TILE_SIZE_X 8
#endif//LOCAL_TILE_SIZE_X

#ifndef LOCAL_TILE_SIZE_Y
  #define LOCAL_TILE_SIZE_Y 8
#endif//LOCAL_TILE_SIZE_Y

/*
2D work group
2D Dispatch
*/

#define WAVEFRONT_SIZE (LOCAL_TILE_SIZE_X*LOCAL_TILE_SIZE_Y)

layout(local_size_x=LOCAL_TILE_SIZE_X,local_size_y=LOCAL_TILE_SIZE_Y)in;
layout(      binding=WRITEDEPTHTEXTURE_BINDING_DEPTH)uniform sampler2DRect Depth;
layout(rg32f,binding=WRITEDEPTHTEXTURE_BINDING_HDT  )uniform image2D       HDT;
layout(      binding=WRITEDEPTHTEXTURE_BINDING_NORMAL)uniform sampler2D  Normal;

uniform uvec2 windowSize;
uniform vec4 lightPosition = vec4(100,100,100,1);

void main(){
  //are we outside of bounds of window?
  if(any(greaterThanEqual(gl_GlobalInvocationID.xy,windowSize)))return;
#if DISCARD_BACK_FACING == 1
  if(dot(lightPosition,texelFetch(Normal,ivec2(gl_GlobalInvocationID.xy),0))>0)
    imageStore(
      HDT,
      ivec2(gl_GlobalInvocationID.xy),
      vec4(texelFetch(Depth,ivec2(gl_GlobalInvocationID.xy)).r*2-1));
  else
    imageStore(
      HDT,
      ivec2(gl_GlobalInvocationID.xy),
      vec4(10));
#else
  imageStore(
    HDT,
    ivec2(gl_GlobalInvocationID.xy),
    vec4(texelFetch(Depth,ivec2(gl_GlobalInvocationID.xy)).r*2-1));
#endif
}).";

const std::string sintorn::hierarchicalDepthSrc = 
GLSL_LINE
R".(
//"methods/Sintorn/hierarchicaldepthtexture.comp";
//DO NOT EDIT ANYTHING BELOW THIS LINE

#ifndef HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT
  #define HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT  0
#endif//HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT

#ifndef HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT
  #define HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT 1
#endif//HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT

#ifndef WAVEFRONT_SIZE
  #define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#ifndef MAX_LEVELS
  #define MAX_LEVELS 4
#endif//MAX_LEVELS


/*
1D WorkGroup
2D Dispatch
*/

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(rg32f,binding=HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT ) readonly uniform image2D HDTInput;
layout(rg32f,binding=HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT)writeonly uniform image2D HDTOutput;

uniform uvec2 WindowSize;
uniform uint  DstLevel;
uniform uvec2 TileDivisibility[MAX_LEVELS];
uniform uvec2 TileSizeInPixels[MAX_LEVELS];

shared float Shared[WAVEFRONT_SIZE*2];

#define DO_NOT_COUNT_WITH_INFINITY

void main(){
  uvec2 LocalCoord=uvec2(gl_LocalInvocationID.x%TileDivisibility[DstLevel+1].x,gl_LocalInvocationID.x/TileDivisibility[DstLevel+1].x);

  vec2 minmax = vec2(10,10);
  if(all(lessThan(gl_WorkGroupID.xy*TileSizeInPixels[DstLevel]+LocalCoord*TileSizeInPixels[DstLevel+1],WindowSize)))
    minmax=imageLoad(HDTInput,ivec2(gl_WorkGroupID.xy*TileDivisibility[DstLevel+1]+LocalCoord)).xy;

	Shared[gl_LocalInvocationID.x               ]=minmax.x;
	Shared[gl_LocalInvocationID.x+WAVEFRONT_SIZE]=minmax.y;
	for(uint threadsPerLevel = WAVEFRONT_SIZE;threadsPerLevel>1;threadsPerLevel>>=1){
		if(gl_LocalInvocationID.x<threadsPerLevel){
      uint halfThreads = threadsPerLevel>>1;
      uint doMax = uint(gl_LocalInvocationID.x>=halfThreads);
			uint BaseIndex=(gl_LocalInvocationID.x&(halfThreads-1))+(doMax*(WAVEFRONT_SIZE));
			float a=Shared[BaseIndex];
			float b=Shared[BaseIndex+halfThreads];
#ifdef  DO_NOT_COUNT_WITH_INFINITY
			if(a>=1)a=b;
			if(b>=1)b=a;
			if(a<=-1)a=b;
			if(b<=-1)b=a;
#endif//DO_NOT_COUNT_WITH_INFINITY
			if((1-2*int(doMax))*(a-b)>=0)Shared[BaseIndex]=b;
		}
	}
	if(gl_LocalInvocationID.x<1)
    imageStore(HDTOutput,ivec2(gl_WorkGroupID.xy),vec4(Shared[0],Shared[WAVEFRONT_SIZE],0,0));
}).";

