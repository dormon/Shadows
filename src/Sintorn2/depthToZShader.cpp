#include <Sintorn2/depthToZShader.h>

std::string const sintorn2::depthToZShader = R".(

#ifndef NEAR
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR


// converts depth (-1,+1) to Z in view-space
float depthToZ(float d){
#ifdef FAR_IS_INFINITE
  return 2.f*NEAR    /(d - 1);
#else
  return 2.f*NEAR*FAR/(d*(FAR-NEAR)-FAR-NEAR);
#endif
}
).";
