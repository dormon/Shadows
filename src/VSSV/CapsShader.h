#pragma once

#include <string>

std::string const vertexShaderSrc = R".(
#line 6
layout(location=0)in vec3 vertexA;
layout(location=1)in vec3 vertexB;
layout(location=2)in vec3 vertexC;

uniform vec4 light = vec4(10,10,10,1);
uniform mat4 mvp   = mat4(1.f);

vec4 transformToFarCap(vec4 P,float farCap){
  return P*(1.0-farCap) + vec4(P.xyz*light.w-light.xyz,0)*farCap;
}

vec4 transformToFarPlane(vec4 P){
  return P.xyww;
}

void main(){
  int multiplicity = currentMultiplicity(vertexA,vertexB,vertexC,light);

  if(multiplicity==0){gl_Position=vec4(0,0,0,1);return;}

  float farCap = float(gl_InstanceID&1);
  uint  vID    = ((farCap>0)?2u-uint(gl_VertexID):uint(gl_VertexID));

  // this has to be swapped if using strips
  vID ^= uint((multiplicity>0) && (vID<2));//swap 0 1

  vec4 P = vec4(vertexA*float(vID==0)+vertexB*float(vID==1)+vertexC*float(vID==2),1.0);

  P = transformToFarCap(P,farCap);
  P = mvp*P;
  P = transformToFarPlane(P);
  gl_Position = P;
}
).";
