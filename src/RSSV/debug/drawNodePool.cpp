#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>

#include <RSSV/debug/drawNodePool.h>

#include <RSSV/getConfigShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/config.h>


using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawNodePool(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      "wavefrontSize"                    ,
      "rssv.method.debug.dump.config"    ,
      "rssv.method.debug.dump.near"      ,
      "rssv.method.debug.dump.far"       ,
      "rssv.method.debug.dump.fovy"      ,
      "rssv.method.debug.wireframe"      ,
      );

  auto const cfg            = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
  auto const wireframe      =  vars.getBool            ("rssv.method.debug.wireframe"      );

  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID;
  }

  ).";

  std::string const fsSrc = R".(
  uniform uint levelToDraw = 0;

  #if WIREFRAME == 0
  in vec3 gNormal;
  #endif

  const vec4 colors[6] = {
    vec4(.1,.1,.1,1)*5,
    vec4(.1,.0,.0,1)*5,
    vec4(.0,.1,.0,1)*5,
    vec4(.0,.0,.1,1)*5,
    vec4(.1,.1,.0,1)*5,
    vec4(.1,.0,.1,1)*5,
  };
  layout(location=0)out vec4 fColor;
  void main(){

    #if WIREFRAME == 1
    fColor = vec4(colors[levelToDraw]);
    #else
    float df = max(dot(normalize(gNormal),normalize(vec3(1,1,1))),0.1f);
    fColor = vec4(colors[levelToDraw]*df);
    #endif
  }
  ).";

  std::string const gsSrc = R".(
#ifndef WARP
#define WARP 64
#endif//WARP
#line 72
#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef TILE_X
#define TILE_X 8
#endif//TILE_X

#ifndef TILE_Y
#define TILE_Y 8
#endif//TILE_Y

#ifndef MIN_Z_BITS
#define MIN_Z_BITS 9
#endif//MIN_Z_BITS

#ifndef NEAR
#define NEAR 1.f
#endif//NEAR

#ifndef FAR
#define FAR 100.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

layout(points)in;

#if WIREFRAME == 1
layout(line_strip,max_vertices=28)out;
#else
layout(triangle_strip,max_vertices=24)out;
out vec3 gNormal;
#endif

flat in uint vId[];

layout(binding=0,std430)buffer NodePool   {uint  nodePool   [];};
layout(binding=1,std430)buffer AABBPool   {float aabbPool   [];};
layout(binding=2,std430)buffer AABBPointer{uint  aabbPointer[];};

uniform mat4 view;
uniform mat4 proj;

uniform mat4 nodeView;
uniform mat4 nodeProj;

uniform int memoryOptim = 0;

uniform uint levelToDraw = 0;

uniform uint drawTightAABB = 0;

uniform uint drawWithPadding = 0;
uniform float zPadding       = 400.f;

void main(){
  uint gId = vId[0];

  float mminX;
  float mmaxX;
  float mminY;
  float mmaxY;
  float mminZ;
  float mmaxZ;

  float startX;
  float startY;
  float endX  ;
  float endY  ;
  float startZ;
  float endZ  ;

  uvec3 xyz = demorton(gId<<(warpBits*(nofLevels-1-levelToDraw)));

  uint bit  = gId & warpMask;
  uint node = gId >> warpBits;

  uint doesNodeExist = nodePool[nodeLevelOffsetInUints[clamp(levelToDraw,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu));
  if(doesNodeExist == 0)return;

  if(memoryOptim == 1){
    uint w = aabbPointer[nodeLevelOffset[clamp(levelToDraw,0u,5u)]+gId+1];
    mminX = aabbPool[w*floatsPerAABB+0];
    mmaxX = aabbPool[w*floatsPerAABB+1];
    mminY = aabbPool[w*floatsPerAABB+2];
    mmaxY = aabbPool[w*floatsPerAABB+3];
    mminZ = aabbPool[w*floatsPerAABB+4];
    mmaxZ = aabbPool[w*floatsPerAABB+5];
  }else{
    mminX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+0];
    mmaxX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+1];
    mminY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+2];
    mmaxY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+3];
    mminZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+4];
    mmaxZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+5];
  }



  startX = -1.f + xyz.x*levelTileSizeClipSpace[nofLevels-1].x;
  startY = -1.f + xyz.y*levelTileSizeClipSpace[nofLevels-1].y;
  endX   = min(startX + levelTileSizeClipSpace[levelToDraw].x,1.f);
  endY   = min(startY + levelTileSizeClipSpace[levelToDraw].y,1.f);
  startZ = CLUSTER_TO_Z(xyz.z                                   );
  endZ   = CLUSTER_TO_Z(xyz.z+(1u<<levelTileBits[levelToDraw].z));

#ifdef FAR_IS_INFINITE
  float e = -1.f;
  float f = -2.f * float(NEAR);
#else
  float e = -float(FAR + NEAR) / float(FAR - NEAR);
  float f = -2.f * float(NEAR) * float(FAR) / float(FAR - NEAR);
#endif

  if(drawWithPadding == 1){
    mminX += - 0.5f/float(WINDOW_X);
    mminY += - 0.5f/float(WINDOW_Y);
    mmaxX += + 0.5f/float(WINDOW_X);
    mmaxY += + 0.5f/float(WINDOW_Y);
    float aaa = (CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(mminZ))+1) - CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(mminZ)))) / zPadding + CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(mminZ)));
    float bbb = CLUSTER_TO_Z(QUANTIZE_Z(DEPTH_TO_Z(mminZ)));
    float ppp = Z_TO_DEPTH(aaa) - Z_TO_DEPTH(bbb);
    mminZ += - ppp;
    mmaxZ += + ppp;
  }

  if(drawTightAABB != 0){
    startX = mminX;
    endX   = mmaxX;

    startY = mminY;
    endY   = mmaxY;

    startZ = DEPTH_TO_Z(mminZ);
    endZ   = DEPTH_TO_Z(mmaxZ);
  }

  vec4 P[8];
  P[0] = vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));
  P[1] = vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));
  P[2] = vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));
  P[3] = vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));
  P[4] = vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));
  P[5] = vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));
  P[6] = vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));
  P[7] = vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));


  mat4 M = proj*view*inverse(nodeView)*inverse(nodeProj);
#if WIREFRAME == 1
  gl_Position = M*P[0];EmitVertex();
  gl_Position = M*P[1];EmitVertex();
  gl_Position = M*P[3];EmitVertex();
  gl_Position = M*P[2];EmitVertex();
  gl_Position = M*P[0];EmitVertex();
  EndPrimitive();

  gl_Position = M*P[4];EmitVertex();
  gl_Position = M*P[5];EmitVertex();
  gl_Position = M*P[7];EmitVertex();
  gl_Position = M*P[6];EmitVertex();
  gl_Position = M*P[4];EmitVertex();
  EndPrimitive();

  gl_Position = M*P[0];EmitVertex();
  gl_Position = M*P[4];EmitVertex();
  EndPrimitive();
  gl_Position = M*P[1];EmitVertex();
  gl_Position = M*P[5];EmitVertex();
  EndPrimitive();
  gl_Position = M*P[2];EmitVertex();
  gl_Position = M*P[6];EmitVertex();
  EndPrimitive();
  gl_Position = M*P[3];EmitVertex();
  gl_Position = M*P[7];EmitVertex();
  EndPrimitive();
#else
  mat4 N = inverse(nodeView)*inverse(nodeProj);
  gNormal = (N*vec4(0,0,-1,0)).xyz;
  gl_Position = M*P[0];EmitVertex();
  gl_Position = M*P[1];EmitVertex();
  gl_Position = M*P[2];EmitVertex();
  gl_Position = M*P[3];EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(1,0,0,0)).xyz;
  gl_Position = M*P[1];EmitVertex();
  gl_Position = M*P[5];EmitVertex();
  gl_Position = M*P[3];EmitVertex();
  gl_Position = M*P[7];EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,0,1,0)).xyz;
  gl_Position = M*P[5];EmitVertex();
  gl_Position = M*P[4];EmitVertex();
  gl_Position = M*P[7];EmitVertex();
  gl_Position = M*P[6];EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(-1,0,0,0)).xyz;
  gl_Position = M*P[4];EmitVertex();
  gl_Position = M*P[0];EmitVertex();
  gl_Position = M*P[6];EmitVertex();
  gl_Position = M*P[2];EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,1,0,0)).xyz;
  gl_Position = M*P[2];EmitVertex();
  gl_Position = M*P[3];EmitVertex();
  gl_Position = M*P[6];EmitVertex();
  gl_Position = M*P[7];EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,-1,0,0)).xyz;
  gl_Position = M*P[4];EmitVertex();
  gl_Position = M*P[5];EmitVertex();
  gl_Position = M*P[0];EmitVertex();
  gl_Position = M*P[1];EmitVertex();
  EndPrimitive();
#endif
}

  ).";

  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      getDebugConfigShader(vars),
      rssv::mortonShader,
      rssv::demortonShader,
      ge::gl::Shader::define("WIREFRAME",(int)wireframe),
      gsSrc);

  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      ge::gl::Shader::define("WIREFRAME",(int)wireframe),
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawNodePoolProgram",
      vs,
      gs,
      fs
      );

}

void drawNodePool(vars::Vars&vars){
  prepareDrawNodePool(vars);

  auto const cfg             = *vars.get<Config>        ("rssv.method.debug.dump.config"          );

  auto const nodeView        = *vars.get<glm::mat4>     ("rssv.method.debug.dump.viewMatrix"      );
  auto const nodeProj        = *vars.get<glm::mat4>     ("rssv.method.debug.dump.projectionMatrix");
  auto const nodePool        =  vars.get<Buffer>        ("rssv.method.debug.dump.nodePool"        );
  auto const aabbPool        =  vars.get<Buffer>        ("rssv.method.debug.dump.aabbPool"        );

  auto const view            = *vars.get<glm::mat4>     ("rssv.method.debug.viewMatrix"           );
  auto const proj            = *vars.get<glm::mat4>     ("rssv.method.debug.projectionMatrix"     );
  auto const levelsToDraw    =  vars.getUint32          ("rssv.method.debug.levelsToDraw"         );
  auto const drawTightAABB   =  vars.getBool            ("rssv.method.debug.drawTightAABB"        );

  auto const memoryOptim     =  vars.getInt32           ("rssv.method.debug.dump.memoryOptim"     );

  auto vao                   =  vars.get<VertexArray>   ("rssv.method.debug.vao"                  );
  auto prg                   =  vars.get<Program>       ("rssv.method.debug.drawNodePoolProgram"  );

  float zPadding             =  vars.addOrGetFloat      ("rssv.method.debug.zPadding",400         );
  auto  drawWithPadding      =  vars.addOrGetBool       ("rssv.method.debug.drawWithPadding"      );

  vao->bind();
  nodePool->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  prg
    ->setMatrix4fv("nodeView"       ,glm::value_ptr(nodeView) )
    ->setMatrix4fv("nodeProj"       ,glm::value_ptr(nodeProj) )
    ->setMatrix4fv("view"           ,glm::value_ptr(view    ) )
    ->setMatrix4fv("proj"           ,glm::value_ptr(proj    ) )
    ->set1ui      ("drawTightAABB"  ,(uint32_t)drawTightAABB  )
    ->set1ui      ("drawWithPadding",(uint32_t)drawWithPadding)
    ->set1f       ("zPadding"       ,(float)zPadding          )
    ->use();

  if(memoryOptim){
    auto aabbPointer = vars.get<Buffer>("rssv.method.debug.dump.aabbPointer");
    aabbPointer->bindBase(GL_SHADER_STORAGE_BUFFER,2);
    prg->set1i("memoryOptim",memoryOptim);
  }


  for(uint32_t l=0;l<cfg.nofLevels;++l){
    if((levelsToDraw&(1u<<l)) == 0)continue;
    prg->set1ui("levelToDraw",l);
    glDrawArrays(GL_POINTS,0,cfg.nofNodesPerLevel[l]);
  }

  vao->unbind();

}

}
