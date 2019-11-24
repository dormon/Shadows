#include<CSSV/capsShaders.h>

std::string const cssv::capsVPSrc = R".(
#version 450 core
layout(location=0)in vec4 Position;
void main(){
  gl_Position=Position;
}).";

std::string const cssv::capsGPSrc = R".(

#define SHIFT_TRIANGLE_TO_INFINITY

#ifdef SHIFT_TRIANGLE_TO_INFINITY
#define SWIZZLE xyww
#else
#define SWIZZLE xyzw
#endif

//front cap, light facing
#define FCLF0 0
#define FCLF1 1
#define FCLF2 2
//back cap, light facing
#define BCLF0 2
#define BCLF1 1
#define BCLF2 0
//front cap, light back-facing
#define FCLB0 2
#define FCLB1 1
#define FCLB2 0
//back cap, light back-facing
#define BCLB0 0
#define BCLB1 1
#define BCLB2 2

layout(triangles)in;
layout(triangle_strip,max_vertices=6)out;

uniform mat4 mvp           = mat4(1.f);
uniform vec4 lightPosition = vec4(100,100,100,1);

void main(){
  int Multiplicity = currentMultiplicity(gl_in[0].gl_Position.xyz,gl_in[1].gl_Position.xyz,gl_in[2].gl_Position.xyz,lightPosition);

  if(Multiplicity==0)return;

  if(Multiplicity>0){
    gl_Position=(mvp*gl_in[FCLF0].gl_Position).SWIZZLE;EmitVertex(); 
    gl_Position=(mvp*gl_in[FCLF1].gl_Position).SWIZZLE;EmitVertex(); 
    gl_Position=(mvp*gl_in[FCLF2].gl_Position).SWIZZLE;EmitVertex(); 
    EndPrimitive();
    //*
    if(lightPosition.w>0){
      gl_Position=(mvp*vec4(gl_in[BCLF0].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      gl_Position=(mvp*vec4(gl_in[BCLF1].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      gl_Position=(mvp*vec4(gl_in[BCLF2].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      EndPrimitive();
    }// */
  }else{
    gl_Position=(mvp*gl_in[FCLB0].gl_Position).SWIZZLE;EmitVertex(); 
    gl_Position=(mvp*gl_in[FCLB1].gl_Position).SWIZZLE;EmitVertex(); 
    gl_Position=(mvp*gl_in[FCLB2].gl_Position).SWIZZLE;EmitVertex(); 
    EndPrimitive();
    //*
    if(lightPosition.w>0){
      gl_Position=(mvp*vec4(gl_in[BCLB0].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      gl_Position=(mvp*vec4(gl_in[BCLB1].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      gl_Position=(mvp*vec4(gl_in[BCLB2].gl_Position.xyz-lightPosition.xyz,0));EmitVertex();
      EndPrimitive();
    }// */
  }
}).";


