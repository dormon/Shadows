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

layout(local_size_x=WORKGROUP_SIZE_X)in;

layout(std430,binding=0)readonly buffer EdgePlanes         {float edgePlanes [];};

layout(std430,binding=3)buffer MultBuffer{
  uint nofSilhouettes;
  uint multBuffer[];
};

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

  if(gl_GlobalInvocationID.x>=NOF_EDGES)return;

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
    globalOffset = atomicAdd(nofSilhouettes,localCounter);
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
