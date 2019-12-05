#include <CSSV/sides/drawShaders.h>

std::string const cssv::sides::drawVPSrc = R".(

#if EXTRACT_MULTIPLICITY == 1
flat out uint vId;
#endif


#if EXTRACT_MULTIPLICITY != 1
layout(location=0)in vec4 Position;
#endif
void main(){
#if EXTRACT_MULTIPLICITY != 1
  gl_Position=Position;
#endif

#if EXTRACT_MULTIPLICITY == 1
  vId = gl_VertexID;
#endif
}).";

std::string const cssv::sides::drawCPSrc = R".(
#version 450 core
layout(vertices=4)out;
uniform mat4 mvp           = mat4(1)            ;
uniform vec4 lightPosition = vec4(100,100,100,1);
void main(){
  //gl_out[gl_InvocationID].gl_Position=mvp*gl_in[gl_InvocationID].gl_Position;
  gl_out[gl_InvocationID].gl_Position=mvp*(vec4(gl_in[gl_InvocationID&1].gl_Position.xyz-lightPosition.xyz*(gl_InvocationID/2),1-(gl_InvocationID/2)));
  if(gl_InvocationID==0){
    gl_TessLevelOuter[0]=1;
    gl_TessLevelOuter[1]=1;
    gl_TessLevelOuter[2]=1;
    gl_TessLevelOuter[3]=1;
    gl_TessLevelInner[0]=1;
    gl_TessLevelInner[1]=1;
  }
}).";

std::string const cssv::sides::drawEPSrc = R".(
#version 450 core
layout(quads)in;
void main(){
  gl_Position = gl_in[uint(gl_TessCoord.x>.5)+(uint(gl_TessCoord.y>.5)<<1)].gl_Position;
}).";

std::string const cssv::sides::drawGPSrc = R".(
layout(points)in;
layout(triangle_strip,max_vertices=8)out;

#if DONT_PACK_MULT == 1
layout(binding=3)buffer MultBuffer{int   multBuffer[];};
#else
layout(binding=3)buffer MultBuffer{uint   multBuffer[];};
#endif
layout(binding=4)buffer EdgeBuffer{float edgeBuffer[];};

flat in uint vId[];

uniform mat4 mvp           = mat4(1)            ;
uniform vec4 lightPosition = vec4(100,100,100,1);

void main(){
#if DONT_PACK_MULT == 1
  int mult = multBuffer[vId[0]*2+0];
  int edge = multBuffer[vId[0]*2+1];
#else
  uint res = multBuffer[vId[0]];
  uint edge = res & 0x1fffffffu;
  int  mult = int((res >> 29u)&0x3u)*int(1-int((res>>31u)<<1u));
#endif
  

  vec4 P[4];
  //P[0] = vec4(edgeBuffer[edge*6+0],edgeBuffer[edge*6+1],edgeBuffer[edge*6+2],1);
  //P[1] = vec4(edgeBuffer[edge*6+3],edgeBuffer[edge*6+4],edgeBuffer[edge*6+5],1);
  P[0][0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
  P[0][1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
  P[0][2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
  P[0][3] = 1;
  P[1][0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
  P[1][1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
  P[1][2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];
  P[1][3] = 1;

  P[2] = vec4(P[0].xyz*lightPosition.w-lightPosition.xyz,0);
  P[3] = vec4(P[1].xyz*lightPosition.w-lightPosition.xyz,0);

  uint swap = uint(mult > 0);
  gl_Position = mvp * P[ +swap];EmitVertex();
  gl_Position = mvp * P[1-swap];EmitVertex();
  gl_Position = mvp * P[2+swap];EmitVertex();
  gl_Position = mvp * P[3-swap];EmitVertex();
  EndPrimitive();
  if(abs(mult) > 1){
    gl_Position = mvp * P[ +swap];EmitVertex();
    gl_Position = mvp * P[1-swap];EmitVertex();
    gl_Position = mvp * P[2+swap];EmitVertex();
    gl_Position = mvp * P[3-swap];EmitVertex();
    EndPrimitive();
  }
}
).";

