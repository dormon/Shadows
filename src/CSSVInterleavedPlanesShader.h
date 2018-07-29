#pragma once

#include<string>

const std::string computeSrc = R".(
#line 6
#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

#ifndef WORKGROUP_SIZE_X
#define WORKGROUP_SIZE_X 64
#endif//WORKGROUP_SIZE_X

//#define CULL_SIDES 1
//#define LOCAL_ATOMIC 1

layout(local_size_x=WORKGROUP_SIZE_X)in;
layout(std430,binding=0)readonly buffer Edges              {float edges      [];};
layout(std430,binding=1)         buffer Silhouettes        {vec4  silhouettes[];};

#if LOCAL_ATOMIC == 1
layout(std430,binding=2)volatile buffer DrawIndirectCommand{uint drawIndirectBuffer[4];};
#else
layout(std430,binding=2)         buffer DrawIndirectCommand{uint drawIndirectBuffer[4];};
#endif

uniform uint numEdge       = 0                  ;
uniform vec4 lightPosition = vec4(100,100,100,1);
uniform mat4 mvp           = mat4(1)            ;

#if LOCAL_ATOMIC == 1
shared uint localCounter;
shared uint globalOffset;
#endif

uint align(uint w,uint a){
  return (w / a) * a + uint((w%a)!=0)*a;
}

void main(){

#if LOCAL_ATOMIC == 1
  if(gl_LocalInvocationID.x==0){
    localCounter = 0;
    globalOffset = 0;
  }
  barrier();
#endif

  uint gid=gl_GlobalInvocationID.x;

  if(gid>=numEdge)return;
  vec4 P[2];

  gid*=3+3+4*MAX_MULTIPLICITY;

  const uint alignSize = 128;
  const uint floatAlignSize = alignSize / 4;

  P[0].x = edges[gl_GlobalInvocationID.x+align(numEdge*0,floatAlignSize)];
  P[0].y = edges[gl_GlobalInvocationID.x+align(numEdge*1,floatAlignSize)];
  P[0].z = edges[gl_GlobalInvocationID.x+align(numEdge*2,floatAlignSize)];
  P[0].w = 1;
  P[1].x = edges[gl_GlobalInvocationID.x+align(numEdge*3,floatAlignSize)];
  P[1].y = edges[gl_GlobalInvocationID.x+align(numEdge*4,floatAlignSize)];
  P[1].z = edges[gl_GlobalInvocationID.x+align(numEdge*5,floatAlignSize)];
  P[1].w = 1;

  precise int Multiplicity=0;

  for(uint m=0;m<MAX_MULTIPLICITY;++m){
    vec4 plane;
    plane.x = edges[gl_GlobalInvocationID.x+align(numEdge*(6+m*4+0),floatAlignSize)];
    plane.y = edges[gl_GlobalInvocationID.x+align(numEdge*(6+m*4+1),floatAlignSize)];
    plane.z = edges[gl_GlobalInvocationID.x+align(numEdge*(6+m*4+2),floatAlignSize)];
    plane.w = edges[gl_GlobalInvocationID.x+align(numEdge*(6+m*4+3),floatAlignSize)];
    Multiplicity += int(sign(dot(plane,lightPosition)));
  }


#if LOCAL_ATOMIC == 1
  uint localOffset = atomicAdd(localCounter,uint(2*abs(Multiplicity)));
  barrier();
  if(gl_LocalInvocationID.x==0){
    globalOffset = atomicAdd(drawIndirectBuffer[0],localCounter);
  }
  barrier();
  uint WH = globalOffset + localOffset;
  if(Multiplicity>0){
    for(int m=0;m<Multiplicity;++m){
      silhouettes[WH++]=P[1];
      silhouettes[WH++]=P[0];
    }
  }
  if(Multiplicity<0){
    Multiplicity=-Multiplicity;
    for(int m=0;m<Multiplicity;++m){
      silhouettes[WH++]=P[0];
      silhouettes[WH++]=P[1];
    }
  }
#else
  if(Multiplicity>0){
    uint WH=atomicAdd(drawIndirectBuffer[0],2*Multiplicity);
    for(int m=0;m<Multiplicity;++m){
      silhouettes[WH++]=P[1];
      silhouettes[WH++]=P[0];
    }
  }
  if(Multiplicity<0){
    Multiplicity=-Multiplicity;
    uint WH=atomicAdd(drawIndirectBuffer[0],2*Multiplicity);
    for(int m=0;m<Multiplicity;++m){
      silhouettes[WH++]=P[0];
      silhouettes[WH++]=P[1];
    }
  }
#endif
}).";

