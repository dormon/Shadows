#pragma once

#include <string>

std::string const vertexShaderSrc = R".(
#line 6

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

layout(location=0)in vec3  edgeVertexA      ;
layout(location=1)in vec3  edgeVertexB      ;
layout(location=2)in float nofOppositeVertices;
layout(location=3)in vec3  oppositeVertices[MAX_MULTIPLICITY];

uniform vec4 lightPosition    = vec4(10,10,10,1);
uniform mat4 modelMatrix      = mat4(1.f);
uniform mat4 viewMatrix       = mat4(1.f);
uniform mat4 projectionMatrix = mat4(1.f);

void main(){
#ifdef USE_TRIANGLE_STRIPS
  int vertexIDCCW = gl_VertexID;
  int vertexIDCW  = gl_VertexID^0x1;
#else
  int vertexIDCCW = int(gl_VertexID>2?6-gl_VertexID:gl_VertexID);
  int vertexIDCW  = int(gl_VertexID>2?gl_VertexID-2:2-gl_VertexID);
#endif//USE_TRIANGLE_STRIPS

  mat4 mvp    = projectionMatrix*viewMatrix*modelMatrix;
  uint sideID = uint(gl_InstanceID%MAX_MULTIPLICITY);
  vec4 P[4];
  P[0]=vec4(edgeVertexA.xyz,1);
  P[1]=vec4(edgeVertexB.xyz,1);
  P[2]=vec4(P[0].xyz*lightPosition.w-lightPosition.xyz,0);
  P[3]=vec4(P[1].xyz*lightPosition.w-lightPosition.xyz,0);
  int multiplicity = 0;

  for(uint m=0;m<uint(nofOppositeVertices);++m)
    multiplicity += currentMultiplicity(P[0].xyz,P[1].xyz,oppositeVertices[m].xyz,lightPosition);
  
  if(sideID >= abs(multiplicity))
    return;

  if(multiplicity>0)
    gl_Position = mvp*P[vertexIDCCW];
  if(multiplicity<0)
    gl_Position = mvp*P[vertexIDCW];
}
).";
