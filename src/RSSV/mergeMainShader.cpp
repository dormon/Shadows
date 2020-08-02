#include <RSSV/mergeMainShader.h>

namespace rssv{
std::string const extern mergeMainShader = R".(

layout(local_size_x=WARP)in;

layout(std430,binding=2)volatile buffer JobCounters       {
  uint silhouetteJobCounter;
  uint triangleJobCounter  ;
  uint traverseDoneCounter ;
  uint mergeJobCounter     ;
  uint dummy               [6];
};

layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

void main(){
  mergeJOB();
}

).";
}
