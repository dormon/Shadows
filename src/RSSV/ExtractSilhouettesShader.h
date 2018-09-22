#include<string>

std::string const extractSilhouettesSrc = R".(
#line 4

#ifndef BUFFER_ALIGNMENT
#define BUFFER_ALIGNMENT 1
#endif//BUFFER_ALIGNMENT

#define FLOAT_SIZE 4

#define FLOAT_BUFFER_ALIGNMENT (BUFFER_ALIGNMENT/FLOAT_SIZE + uint(BUFFER_ALIGNMENT%FLOAT_SIZE != 0))

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

#ifndef WGS
#define WGS 64
#endif//WGS

layout(local_size_x=WGS)in;

layout(std430,binding=0)readonly buffer Edges                 {float edges                 [ ];};
layout(std430,binding=1)         buffer Silhouettes           {float silhouettes           [ ];};
layout(std430,binding=2)         buffer DispatchIndirectBuffer{uint  dispatchIndirectBuffer[3];};

uniform uint numEdge       = 0;
uniform vec4 lightPosition = vec4(10,10,10,1);

shared uint globalOffset;

uint align(uint w,uint a){
  return (w / a) * a + uint((w%a)!=0)*a;
}

void main(){
  uint gid=gl_GlobalInvocationID.x;
  if(gid >= numEdge)return;

  precise int multiplicity=0;
  for(uint m=0;m<MAX_MULTIPLICITY;++m){
    vec4 plane;
    plane.x = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+0)];
    plane.y = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+1)];
    plane.z = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+2)];
    plane.w = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*(6+m*4+3)];
    multiplicity += int(sign(dot(plane,lightPosition)));
  }

  UINT_RESULT_ARRAY nonZeroMultiplicity = TRANSFORM_BALLOT_RESULT_TO_UINTS(BALLOT(abs(multiplicity) != 0));
  if(gl_LocalInvocationID.x == 0){
    uint nofSilhouettes = 0;
    for(uint i=0;i<BALLOT_RESULT_LENGTH;++i)
      nofSilhouettes += bitCount(GET_UINT_FROM_UINT_ARRAY(nonZeroMultiplicity,i));
    globalOffset = atomicAdd(dispatchIndirectBuffer[0],nofSilhouettes);
  }

  uint localOffset = 0;
  for(uint i=0;i<BALLOT_RESULT_LENGTH;++i){
    uint mask = 0;
    if(gl_LocalInvocationID.x >= (i+1)*32)
      mask = 0xffffffffu;
    else if(gl_LocalInvocationID.x >= i*32)
      mask = (1<<(gl_LocalInvocationID.x - i*32)) - 1;
    else
      mask = 0x00000000u;
    localOffset += bitCount(GET_UINT_FROM_UINT_ARRAY(nonZeroMultiplicity,i)&mask);
  }
    
  uint offset = (globalOffset + localOffset);

  float P[6];
  for(uint i=0;i<6;++i)
    P[i] = edges[gl_GlobalInvocationID.x+align(numEdge,FLOAT_BUFFER_ALIGNMENT)*i];

  if(multiplicity == 0)return;
  for(uint i=0;i<6;++i)
    silhouettes[offset + i] = P[(i+3*uint(multiplicity>0))%6];
  silhouettes[offset + 6] = float(abs(multiplicity));
}
).";
