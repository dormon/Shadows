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

std::string const sintorn::writeStencilShader = 
GLSL_LINE
R".(
//"methods/Sintorn/writestenciltexture.comp";
//DO NOT EDIT ANYTHING BELOW THIS LINE

#ifndef WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK
  #define WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK 0
#endif//WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK

#ifndef WRITESTENCILTEXTURE_BINDING_HSTINPUT
  #define WRITESTENCILTEXTURE_BINDING_HSTINPUT 1
#endif//WRITESTENCILTEXTURE_BINDING_HSTINPUT

#ifndef LOCAL_TILE_SIZE_X
  #define LOCAL_TILE_SIZE_X 8
#endif//LOCAL_TILE_SIZE_X

#ifndef LOCAL_TILE_SIZE_Y
  #define LOCAL_TILE_SIZE_Y 8
#endif//LOCAL_TILE_SIZE_Y

#define SHADOW_VALUE               1
#define WAVEFRONT_SIZE             (LOCAL_TILE_SIZE_X*LOCAL_TILE_SIZE_Y)
#define UINT_BIT_SIZE              32
#define RESULT_LENGTH_IN_UINT      (WAVEFRONT_SIZE/UINT_BIT_SIZE)
#define RESULT_ID                  (INVOCATION_ID_IN_WAVEFRONT/UINT_BIT_SIZE)
#define INVOCATION_ID_IN_WAVEFRONT (gl_LocalInvocationID.y*LOCAL_TILE_SIZE_X+gl_LocalInvocationID.x)

layout(local_size_x=LOCAL_TILE_SIZE_X,local_size_y=LOCAL_TILE_SIZE_Y)in;

layout(r32ui,binding=WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK)uniform uimage2D FinalStencilMask;
layout(r32ui,binding=WRITESTENCILTEXTURE_BINDING_HSTINPUT        )uniform uimage2D HSTInput;

uniform uvec2 WindowSize;

void main(){
  //are we outside of bounds of window?
  if(any(greaterThanEqual(gl_GlobalInvocationID.xy,WindowSize)))return;
  uint stencilValue=imageLoad(HSTInput,ivec2(gl_WorkGroupID.x*RESULT_LENGTH_IN_UINT+RESULT_ID,gl_WorkGroupID.y)).x;
  if(((stencilValue>>(INVOCATION_ID_IN_WAVEFRONT%UINT_BIT_SIZE))&1u)!=0u&&gl_GlobalInvocationID.y>=8*16*0)
    imageStore(FinalStencilMask,ivec2(gl_GlobalInvocationID),uvec4(SHADOW_VALUE));
}).";

