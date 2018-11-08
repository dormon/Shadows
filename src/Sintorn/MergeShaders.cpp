#include <Sintorn/MergeShaders.h>
#include <GLSLLine.h>

std::string const sintorn::mergeShader = 
GLSL_LINE
R".(
//"methods/Sintorn/mergetexture.comp";
//DO NOT EDIT ANYTHING BELOW THIS LINE

#ifndef MERGETEXTURE_BINDING_HSTINPUT
#define MERGETEXTURE_BINDING_HSTINPUT 0
#endif//MERGETEXTURE_BINDING_HSTINPUT

#ifndef MERGETEXTURE_BINDING_HSTOUTPUT
#define MERGETEXTURE_BINDING_HSTOUTPUT 1
#endif//MERGETEXTURE_BINDING_HSTOUTPUT

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#define UINT_BIT_SIZE              32
#define RESULT_LENGTH_IN_UINT      (WAVEFRONT_SIZE/UINT_BIT_SIZE)
#define INVOCATION_ID_IN_WAVEFRONT (uint(gl_LocalInvocationID.x))

uniform uvec2 WindowSize;
uniform uvec2 DstTileSizeInPixels;
uniform uvec2 DstTileDivisibility;

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(r32ui,binding=MERGETEXTURE_BINDING_HSTINPUT ) readonly uniform uimage2D HSTInput;
layout(r32ui,binding=MERGETEXTURE_BINDING_HSTOUTPUT)writeonly uniform uimage2D HSTOutput;

void main(){
  ivec2  LocalCoord = ivec2(INVOCATION_ID_IN_WAVEFRONT%DstTileDivisibility.x,INVOCATION_ID_IN_WAVEFRONT/DstTileDivisibility.x);
  ivec2 GlobalCoord = ivec2(gl_WorkGroupID.xy*DstTileDivisibility+LocalCoord);

  if(any(greaterThanEqual(GlobalCoord*DstTileSizeInPixels,WindowSize)))return;

  uint stencilValue=imageLoad(HSTInput,ivec2(gl_WorkGroupID.x*RESULT_LENGTH_IN_UINT+(INVOCATION_ID_IN_WAVEFRONT/UINT_BIT_SIZE),gl_WorkGroupID.y)).x;
  if(((stencilValue>>(INVOCATION_ID_IN_WAVEFRONT%UINT_BIT_SIZE))&1u)!=0u)
    for(uint r=0;r<RESULT_LENGTH_IN_UINT;++r)
      imageStore(HSTOutput,ivec2(GlobalCoord.x*RESULT_LENGTH_IN_UINT+r,GlobalCoord.y),uvec4(0xffffffffu));
}).";
