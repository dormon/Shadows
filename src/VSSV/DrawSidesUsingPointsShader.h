#pragma once

#include <string>

std::string const vertexShaderSrc = R".(
#line 6

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

layout(location=0)in vec3  vertexA      ;
layout(location=1)in vec3  vertexB      ;
layout(location=2)in float nofOppositeVertices;
layout(location=3)in vec3  oppositeVertices[MAX_MULTIPLICITY];

uniform vec4 light = vec4(10,10,10,1);
uniform mat4 mvp   = mat4(1.f);

void main(){
#ifdef USE_TRIANGLE_STRIPS
  uint vertexIDCCW = gl_VertexID;
  uint vertexIDCW  = gl_VertexID^0x1;
#else
  uint vertexIDCCW = int(gl_VertexID>2?6-gl_VertexID:gl_VertexID);
  uint vertexIDCW  = int(gl_VertexID>2?gl_VertexID-2:2-gl_VertexID);
#endif//USE_TRIANGLE_STRIPS

  uint sideID = uint(gl_InstanceID%MAX_MULTIPLICITY);
  vec4 P[4];
  P[0] = vec4(vertexA.xyz,1);
  P[1] = vec4(vertexB.xyz,1);
  P[2] = vec4(P[0].xyz*light.w-light.xyz,0);
  P[3] = vec4(P[1].xyz*light.w-light.xyz,0);
  int multiplicity = 0;

  for(uint m=0;m<uint(nofOppositeVertices);++m)
    multiplicity += currentMultiplicity(P[0].xyz,P[1].xyz,oppositeVertices[m].xyz,light);
  
  if(sideID >= abs(multiplicity))
    return;

  if(multiplicity>0)
    gl_Position = mvp*P[vertexIDCCW];
  if(multiplicity<0)
    gl_Position = mvp*P[vertexIDCW ];
}
).";
