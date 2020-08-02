#include <RSSV/globalBarrierShader.h>

namespace rssv{
std::string const globalBarrierShaderFWD = R".(
void globalBarrier();
).";

std::string const globalBarrierShader = R".(
void globalBarrier(){
  //every WGS increments counter
  if(gl_LocalInvocationIndex == 0){
    atomicAdd(traverseDoneCounter,1);
  }

  //and wait until all WGS finish
  for(int i=0;i<1000;++i){
    //first thread read number of finished WGS
    uint finishedWGS;
    if(gl_LocalInvocationIndex == 0)
      finishedWGS = traverseDoneCounter;

    finishedWGS = readFirstInvocationARB(finishedWGS);
    if(finishedWGS == gl_NumWorkGroups.x)break;

    //active waiting to prevent excesive global memory read...
    if(gl_LocalInvocationIndex == 0){
      uint c=0;
      for(uint j=0;j<100;++j)
        c = (c+finishedWGS+j)%177;
      if(c == 1337)dummy[2] = 1111;
    }
  }
}
).";

}
