#pragma once

#include<iostream>

const std::string convertStencilBufferToShadowMaskVPSrc = R".(
#version 450 core
void main(){
  gl_Position = vec4(-1+2*(gl_VertexID>>1),-1+2*(gl_VertexID&1),0,1);
}
).";

const std::string convertStencilBufferToShadowMaskFPSrc = R".(
#version 450 core
layout(location=0)out float fColor;
void main(){
  fColor = 1;
}
).";

