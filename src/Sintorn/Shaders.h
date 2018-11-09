#pragma once

#include<iostream>

const std::string blitCompSrc = R".(
#version 450 core
layout(local_size_x=8,local_size_y=8)in;

uniform uvec2 windowSize = uvec2(512,512);

layout(r32ui,binding=0)readonly  uniform uimage2D finalStencilMask;
layout(r32f ,binding=1)writeonly uniform  image2D shadowMask      ;

void main(){
  if(any(greaterThanEqual(gl_GlobalInvocationID.xy,windowSize)))return;
  uint S=imageLoad(finalStencilMask,ivec2(gl_GlobalInvocationID.xy)).x;
  imageStore(shadowMask,ivec2(gl_GlobalInvocationID.xy),vec4(1-S));
}
).";


const std::string drawHSTVertSrc = R".(
#version 450 core
void main(){
  gl_Position = vec4(-1+2*(gl_VertexID%2),-1+2*(gl_VertexID/2),0,1);
}
).";

const std::string drawHSTFragSrc = R".(
#version 450 core
#define UINT_BIT_SIZE 32

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#define RESULT_LENGTH_IN_UINT         (WAVEFRONT_SIZE/UINT_BIT_SIZE)

layout(location=0)out vec4 fColor;
layout(r32ui,binding=0)readonly uniform uimage2D HST;

uvec2 Coord=uvec2(gl_FragCoord.xy);

uniform uvec2 windowSize = uvec2(512,512);

void main(){
  Coord = uvec2(vec2(Coord)*imageSize(HST)*vec2(0.5,1)/windowSize);
  uvec2 cc=(Coord)%uvec2(8);
  uint invocationId=cc.y*8+cc.x;
  uint stencilValue=imageLoad(HST,ivec2((Coord.x/8)*RESULT_LENGTH_IN_UINT+(invocationId/UINT_BIT_SIZE),(Coord.y/8))).x;
  uint shadow=(stencilValue>>(invocationId%UINT_BIT_SIZE))&1u;
  fColor=vec4(1-shadow,1,0,1);
  return;
}
).";

const std::string drawFinalStencilMaskFragSrc = R".(
#version 450 core
#define UINT_BIT_SIZE 32

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#define RESULT_LENGTH_IN_UINT         (WAVEFRONT_SIZE/UINT_BIT_SIZE)

layout(location=0)out vec4 fColor;
layout(r32ui,binding=0)readonly uniform uimage2D FinalStencilMask;

uvec2 Coord=uvec2(gl_FragCoord.xy);

void main(){
  fColor=vec4(1,0,0,1);
  uint S=imageLoad(FinalStencilMask,ivec2(Coord)).x;
  fColor=vec4(1-S,1,0,1);
}
).";
