#include <RSSV/loadEdgeShader.h>

std::string const rssv::loadEdgeShaderFWD = R".(

#define ALIGN(W,A) uint(uint(uint(W)/uint(A))*uint(A) + uint((uint(W)%uint(A))!=0u)*uint(A))
#define ALIGN_SIZE_FLOAT ALIGN(ALIGN_SIZE,4u)
#define ALIGN_OFFSET(i) uint(ALIGN(NOF_EDGES,ALIGN_SIZE_FLOAT)*uint(i))

void loadEdge(out vec3 A,out vec3 B,uint edge);
).";
std::string const rssv::loadEdgeShader = R".(

void loadEdge(out vec3 A,out vec3 B,uint edge){
  A[0] = edgePlanes[edge+ALIGN_OFFSET(0)];
  A[1] = edgePlanes[edge+ALIGN_OFFSET(1)];
  A[2] = edgePlanes[edge+ALIGN_OFFSET(2)];
  B[0] = edgePlanes[edge+ALIGN_OFFSET(3)];
  B[1] = edgePlanes[edge+ALIGN_OFFSET(4)];
  B[2] = edgePlanes[edge+ALIGN_OFFSET(5)];
}

).";
