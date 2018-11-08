#include <Sintorn/HierarchyShaders.h>
#include <GLSLLine.h>

const std::string sintorn::writeDepth = 
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


