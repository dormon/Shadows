#include <RSSV/PerfectResolution/Build.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Deferred.h>
#include <glm/glm.hpp>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace rssv;

PerfectHierarchy::PerfectHierarchy(vars::Vars&vars):BuildHierarchy(vars){
  auto const windowSize    = vars.get<uvec2>("windowSize"    );
  auto const wavefrontSize = vars.getSizeT  ("wavefrontSize" );
  assert(wavefrontSize != 64);
  assert(windowSize->x == windowSize->y && windowSize->x == 512);

  vars.add<Texture>("level0",GL_TEXTURE_2D,GL_RG32F,8 ,8 );
  vars.add<Texture>("level1",GL_TEXTURE_2D,GL_RG32F,64,64);

  string const l1src = R".(
  layout(local_size_x=64)in;
  layout(      binding=0)uniform sampler2D depth ;
  layout(rg32f,binding=1)uniform image2D   level1;

  float getDepth(uvec2 coord){
    return texelFeth(depth,ivec2(coord),0).x * 2 - 1;
  }

  float localDepth[64];
  void main(){
    uint parentId     = gl_WorkGroupID.x;
    uvec2 parentCoord = uvec2(parent&7,parent>>3);
    uint localId      = gl_LocalInvocationID.x;
    uvec2 localCoord  = uvec2(localId&7,parent>>3);
    uvec2 parentSize  = uvec2(8,8);
    uvec2 globalCoord = parentCoord*parentSize + localCoord;
    localDepth[gl_LocalInvocationID.x] = getDepth(globalCoord);

    if(localId < 32){
      float a = localDepth[localId+0 ];
      float b = localDepth[localId+32];
      localDepth[localId+0 ] = min(a,b);
      localDepth[localId+32] = max(a,b);
    }
    for(uint halfSize = 32;halfSize > 1;halfSize>>=1){
      if(localId >= halfSize)continue;
      uint quarter = halfSize>>1;
      uint doMax = uint(localId >= quarter);
      float a = localDepth[localId + 0       + doMax*halfSize];
      float b = localDepth[localId + quarter + doMax*halfSize];
      uint bIsLess = uint(b-a < 0);
      uint needToSwap = bIsLess^doMax;
      localDepth[localId + doMax*quarter] = a*(1-needToSwap) + b*needToSwap;
    }
    if(localId == 0){
      uint oddLine = uint(parentCoord.y&1);
      uint last = (parentCoord.x == 63)*(1-oddLine) + (parentCoord.x == 0)*oddLine;
      //uvec2(parentCoord.x+1,parentCoord.y+0)*parentSize + uvec2(0,0) 
      //uvec2(parentCoord.x+0,parentCoord.y+1)*parentSize + uvec2(0,0)
      //uvec2(parentCoord.x-1,parentCoord.y+0)*parentSize + uvec2(7,7)
      
      uvec2 bridgeCoord = uvec2(parentCoord.x+(1-2*oddLine)*(1-last)),parentCoord.y+last)*parentSize + uvec2(7*(1-last)*oddLine);
      float bridge = getDepth(bridgeCoord);

      imageStore(level1,ivec2(parentCoord),vec2(min(bridge,localDepth[0]),max(bridge,localDepth[1])));

    }
  }

  ).";
  vars.add<Program>("computeLevel1",make_shared<Shader>(GL_COMPUTE_SHADER,l1src));

  string const l0src = R".(
  layout(local_size_x=64)in;
  layout(rg32f,binding=0)uniform image2D level1;
  layout(rg32f,binding=1)uniform image2D level0;

  float localDepth[64*2];

  vec2 getDepth(uvec2 coord){
    return imageLoad(level1,ivec2(coord)).xy;
  }
  
  void storeDepth(uvec2 coord,vec2 d){
    imageStore(level0,ivec2(coord),vec4(d));
  }

  uvec2 getParentCoord(){
    return uvec2(gl_WorkGroupID.x&7,gl_WorkGroupID.x>>3);
  }

  uvec2 getLocalCoord(){
    return uvec2(gl_LocalInvocationID.x&7,gl_LocalInvocationID.x>>3);
  }

  uvec2 getGlobalCoord(){
    uvec2 parentCoord = getParentCoord();
    uvec2 localCoord  = getLocalCoord();
    uvec2 parentSize  = uvec2(8,8);
    uvec2 globalCoord = parentCoord*parentSize + localCoord;
  }

  void loadToLocal(){
    uvec2 globalCoord = getGlobalCoord();
    vec2 minMax = getDepth(globalCoord);
    localDepth[gl_LocalInvocationID.x+0 ] = minMax.x;
    localDepth[gl_LocalInvocationID.x+64] = minMax.y;
  }

  void storeToGlobal(){
    if(gl_LocalInvocationID.x == 0)
      storeDepth(getParentCoord(),vec2(localDepth[0],localDepth[1]));
  }


  void main(){

    loadToLocal();

    uint localId = gl_LocalInvocationID.x;
    for(uint halfSize = 64;halfSize > 1;halfSize>>=1){
      if(localId >= halfSize)continue;
      uint quarter = halfSize>>1;
      uint doMax = uint(localId >= quarter);
      float a = localDepth[localId + 0       + doMax*halfSize];
      float b = localDepth[localId + quarter + doMax*halfSize];
      uint bIsLess = uint(b-a < 0);
      uint needToSwap = bIsLess^doMax;
      localDepth[localId + doMax*quarter] = a*(1-needToSwap) + b*needToSwap;
    }

    storeToGlobal();

  }

  ).";
  vars.add<Program>("computeLevel0",make_shared<Shader>(GL_COMPUTE_SHADER,l0src));
}

void PerfectHierarchy::build(){
  vars.get<GBuffer>("gBuffer")->depth->bind(0);
  vars.get<Texture>("level1")->bindImage(1);
  vars.get<Program>("computeLevel1")->dispatch(64*64,1,1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

  vars.get<Texture>("level1")->bindImage(0);
  vars.get<Texture>("level0")->bindImage(1);
  vars.get<Program>("computeLevel0")->dispatch(64,1,1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
