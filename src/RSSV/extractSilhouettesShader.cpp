#include <RSSV/extractSilhouettesShader.h>

std::string const rssv::extractSilhouettesShader = R".(
#line 4


#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

#ifndef WORKGROUP_SIZE_X
#define WORKGROUP_SIZE_X 64
#endif//WORKGROUP_SIZE_X

#ifndef ALIGN_SIZE
#define ALIGN_SIZE 128
#endif//ALIGN_SIZE

#ifndef NOF_EDGES
#define NOF_EDGES 0
#endif//NOF_EDGES

#define ALIGN(W,A) uint(uint(uint(W)/uint(A))*uint(A) + uint((uint(W)%uint(A))!=0u)*uint(A))

#define ALIGN_SIZE_FLOAT ALIGN(ALIGN_SIZE,4u)

#define ALIGN_OFFSET(i) uint(ALIGN(NOF_EDGES,ALIGN_SIZE_FLOAT)*uint(i))

layout(local_size_x=WORKGROUP_SIZE_X)in;

layout(std430,binding=0)readonly buffer EdgePlanes         {float edgePlanes [];};

layout(std430,binding=2)volatile buffer DrawIndirectBuffer{uint drawIndirectBuffer[4];};

layout(std430,binding=3)buffer MultBuffer{uint multBuffer[];};

uniform vec4 lightPosition = vec4(100,100,100,1);
uniform mat4 mvp           = mat4(1)            ;

shared uint localCounter;
shared uint globalOffset;

void main(){

  if(gl_LocalInvocationID.x==0){
    localCounter = 0;
    globalOffset = 0;
  }
  barrier();

  uint gid=gl_GlobalInvocationID.x;

  if(gid>=NOF_EDGES)return;

  vec4 P[2];

  gid*=3+3+4*MAX_MULTIPLICITY;

  P[0].x = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(0)];
  P[0].y = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(1)];
  P[0].z = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(2)];
  P[0].w = 1;
  P[1].x = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(3)];
  P[1].y = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(4)];
  P[1].z = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(5)];
  P[1].w = 1;
  
  int Num=int(P[0].w)+2;
  P[0].w=1;

  precise int Multiplicity=0;

  for(uint m=0;m<MAX_MULTIPLICITY;++m){
    vec4 plane;
    plane.x = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(6+m*4+0)];
    plane.y = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(6+m*4+1)];
    plane.z = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(6+m*4+2)];
    plane.w = edgePlanes[gl_GlobalInvocationID.x+ALIGN_OFFSET(6+m*4+3)];
    Multiplicity += int(sign(dot(plane,lightPosition)));
  }

  uint localOffset = atomicAdd(localCounter,uint(Multiplicity!=0));

  #if WORKGROUP_SIZE_X > WARP
    barrier();
  #endif

  if(gl_LocalInvocationID.x==0){
    globalOffset = atomicAdd(drawIndirectBuffer[0],localCounter);
  }

  #if WORKGROUP_SIZE_X > WARP
    barrier();
  #endif

  uint WH = globalOffset + localOffset;
  
  if(Multiplicity != 0){
    uint res = 0;
    res |= uint(Multiplicity << 29);
    res |= uint(gl_GlobalInvocationID.x);
    multBuffer[WH] = res;
  }
}
).";
