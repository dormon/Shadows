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

#include <RSSV/debug/drawBridges.h>

#include <RSSV/mortonShader.h>
#include <RSSV/configShader.h>
#include <RSSV/config.h>


using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawBridges(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      "wavefrontSize"                    ,
      "rssv.method.debug.dump.config"    ,
      "rssv.method.debug.dump.near"      ,
      "rssv.method.debug.dump.far"       ,
      "rssv.method.debug.dump.fovy"      ,
      );

  auto const cfg            = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
  auto const nnear          =  vars.getFloat           ("rssv.method.debug.dump.near"      );
  auto const ffar           =  vars.getFloat           ("rssv.method.debug.dump.far"       );
  auto const fovy           =  vars.getFloat           ("rssv.method.debug.dump.fovy"      );

  auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"                    );


  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID;
  }

  ).";

  std::string const fsSrc = R".(
  uniform uint levelToDraw = 0;

  in vec3 gNormal;

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

    fColor = vec4(colors[levelToDraw]);
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
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

layout(points)in;

layout(line_strip,max_vertices=28)out;

flat in uint vId[];

layout(binding=0)buffer NodePool{uint nodePool[];};
layout(binding=1)buffer AABBPool{float aabbPool[];};

uniform mat4 view;
uniform mat4 proj;

uniform mat4 nodeView;
uniform mat4 nodeProj;

uniform uint levelToDraw = 0;

uniform uint drawTightAABB = 0;

uniform uint drawWithPadding = 0;
uniform float zPadding       = 400.f;


vec3 getCenter(uint level,uint node){
  vec3 minCorner;
  vec3 maxCorner;

  minCorner.x = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+0];
  maxCorner.x = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+1];
  minCorner.y = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+2];
  maxCorner.y = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+3];
  minCorner.z = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+4];
  maxCorner.z = aabbPool[aabbLevelOffsetInFloats[level]+node*floatsPerAABB+5];

  vec3 center = (maxCorner+minCorner)/2.f;
}

void main(){
  uint gId = vId[0];

  uvec3 xyz = demorton(gId<<(warpBits*(nofLevels-1-levelToDraw)));

  uint bit  = gId & warpMask;
  uint node = gId >> warpBits;

  uint doesNodeExist = nodePool[nodeLevelOffsetInUints[clamp(levelToDraw,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu));
  if(doesNodeExist == 0)return;


  mat4 nodeProjView = inverse(nodeView)*inverse(nodeProj);

  vec4 center = nodeProjView*vec4(getCenter(clamp(levelToDraw,0u,5u),gId),1);
  vec4 parentCenter;
  if(levelToDraw == 0){
    parentCenter = lightPosition;
  }else{
    parentCenter = nodeProjView*vec4(getCenter(clamp(levelToDraw-1,0u,5u),gId>>warpBits),1);
  }

  center4       /= center4      .w;
  parentCenter4 /= parentCenter4.w;

  mat4 M = proj*view;

  gl_Position = M*vec4(center4      );EmitVertex();
  gl_Position = M*vec4(parentCenter4);EmitVertex();
  EndPrimitive();

}

  ).";

  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      ge::gl::Shader::define("WARP"      ,(uint32_t)wavefrontSize),
      ge::gl::Shader::define("WINDOW_X"  ,(uint32_t)cfg.windowX  ),
      ge::gl::Shader::define("WINDOW_Y"  ,(uint32_t)cfg.windowY  ),
      ge::gl::Shader::define("MIN_Z_BITS",(uint32_t)cfg.minZBits ),
      ge::gl::Shader::define("NEAR"      ,nnear                  ),
      glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
      ge::gl::Shader::define("FOVY"      ,fovy                   ),
      ge::gl::Shader::define("TILE_X"    ,cfg.tileX              ),
      ge::gl::Shader::define("TILE_Y"    ,cfg.tileY              ),

      rssv::configShader,
      rssv::mortonShader,
      rssv::demortonShader,
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawBridgesProgram",
      vs,
      gs,
      fs
      );

}

void drawBridges(vars::Vars&vars){
  prepareDrawBridges(vars);

  auto const cfg             = *vars.get<Config>        ("rssv.method.debug.dump.config"          );

  auto const nodeView        = *vars.get<glm::mat4>     ("rssv.method.debug.dump.viewMatrix"      );
  auto const nodeProj        = *vars.get<glm::mat4>     ("rssv.method.debug.dump.projectionMatrix");
  auto const nodePool        =  vars.get<Buffer>        ("rssv.method.debug.dump.nodePool"        );
  auto const aabbPool        =  vars.get<Buffer>        ("rssv.method.debug.dump.aabbPool"        );

  auto const view            = *vars.get<glm::mat4>     ("rssv.method.debug.viewMatrix"           );
  auto const proj            = *vars.get<glm::mat4>     ("rssv.method.debug.projectionMatrix"     );
  auto const bridgesToDraw   =  vars.getUint32          ("rssv.method.debug.bridgesToDraw"        );
  auto const drawTightAABB   =  vars.getBool            ("rssv.method.debug.drawTightAABB"        );

  float zPadding = vars.addOrGetFloat("rssv.method.debug.zPadding",400);
  auto  drawWithPadding = vars.addOrGetBool("rssv.method.debug.drawWithPadding");

  auto vao = vars.get<VertexArray>("rssv.method.debug.vao");

  auto prg = vars.get<Program>("rssv.method.debug.drawBridgesProgram");

  vao->bind();
  nodePool->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  prg->use();
  prg
    ->setMatrix4fv("nodeView"   ,glm::value_ptr(nodeView))
    ->setMatrix4fv("nodeProj"   ,glm::value_ptr(nodeProj))
    ->setMatrix4fv("view"       ,glm::value_ptr(view    ))
    ->setMatrix4fv("proj"       ,glm::value_ptr(proj    ))
    ->set1ui      ("drawTightAABB",(uint32_t)drawTightAABB)
    ->set1ui      ("drawWithPadding",(uint32_t)drawWithPadding)
    ->set1f       ("zPadding"       ,(float)zPadding)
    ;


  for(uint32_t l=0;l<cfg.nofLevels;++l){
    if((bridgesToDraw&(1u<<l)) == 0)continue;
    prg->set1ui      ("levelToDraw",l);
    glDrawArrays(GL_POINTS,0,cfg.nofNodesPerLevel[l]);
  }

  vao->unbind();

}

}
