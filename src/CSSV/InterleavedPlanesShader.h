#pragma once

#include<string>

const std::string computeSrc = R".(
#line 6

#ifndef BUFFER_ALIGNMENT
#define BUFFER_ALIGNMENT 1
#endif//BUFFER_ALIGNMENT

#define FLOAT_SIZE 4

#define FLOAT_BUFFER_ALIGNMENT (BUFFER_ALIGNMENT/FLOAT_SIZE + uint(BUFFER_ALIGNMENT%FLOAT_SIZE != 0))

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

#ifndef WORKGROUP_SIZE_X
#define WORKGROUP_SIZE_X 64
#endif//WORKGROUP_SIZE_X

layout(local_size_x=WORKGROUP_SIZE_X)in;
layout(std430,binding=0)readonly buffer Edges              {float edges      [];};
layout(std430,binding=1)         buffer Silhouettes        {vec4  silhouettes[];};
layout(std430,binding=2)volatile buffer DrawIndirectCommand{uint drawIndirectBuffer[4];};

uniform uint numEdge       = 0                  ;
uniform vec4 lightPosition = vec4(100,100,100,1);

uint align(uint w,uint a){
  return (w / a) * a + uint((w%a)!=0)*a;
}


#if LOCAL_ATOMIC == 1
shared uint localCounter;
shared uint globalOffset;
#endif

void main(){
  if(gl_GlobalInvocationID.x>=numEdge)return;

#if LOCAL_ATOMIC == 1
  if(gl_LocalInvocationID.x==0){
    localCounter = 0;
    globalOffset = 0;
  }
#endif


  vec4 P[2];

  P[0].x = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*0];
  P[0].y = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*1];
  P[0].z = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*2];
  P[0].w = 1;
  P[1].x = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*3];
  P[1].y = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*4];
  P[1].z = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*5];
  P[1].w = 1;

  precise int multiplicity=0;

  for(uint m=0;m<MAX_MULTIPLICITY;++m){
    vec4 plane;
    plane.x = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+0)];
    plane.y = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+1)];
    plane.z = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+2)];
    plane.w = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+3)];
    multiplicity += int(sign(dot(plane,lightPosition)));
  }

#if LOCAL_ATOMIC == 1
  barrier();

  uint localOffset = atomicAdd(localCounter,uint(2*abs(multiplicity)));
  barrier();
  if(gl_LocalInvocationID.x==0){
    globalOffset = atomicAdd(drawIndirectBuffer[0],localCounter);
  }
  barrier();
  uint WH = globalOffset + localOffset;
  if(multiplicity>0){
    for(int m=0;m<multiplicity;++m){
      silhouettes[WH++]=P[1];
      silhouettes[WH++]=P[0];
    }
  }
  if(multiplicity<0){
    multiplicity=-multiplicity;
    for(int m=0;m<multiplicity;++m){
      silhouettes[WH++]=P[0];
      silhouettes[WH++]=P[1];
    }
  }
#else
  if(multiplicity>0){
    uint WH=atomicAdd(drawIndirectBuffer[0],2*multiplicity);
    for(int m=0;m<multiplicity;++m){
      silhouettes[WH++]=P[1];
      silhouettes[WH++]=P[0];
    }
  }
  if(multiplicity<0){
    multiplicity=-multiplicity;
    uint WH=atomicAdd(drawIndirectBuffer[0],2*multiplicity);
    for(int m=0;m<multiplicity;++m){
      silhouettes[WH++]=P[0];
      silhouettes[WH++]=P[1];
    }
  }
#endif
}).";

