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

#include <RSSV/mortonShader.h>
#include <RSSV/configShader.h>
#include <RSSV/config.h>


using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawTraverse(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      "wavefrontSize"                        ,
      "rssv.method.debug.dump.config"    ,
      "rssv.method.debug.dump.near"      ,
      "rssv.method.debug.dump.far"       ,
      "rssv.method.debug.dump.fovy"      ,
      "rssv.method.debug.wireframe"      ,
      );

  auto const cfg            = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
  auto const nnear          =  vars.getFloat           ("rssv.method.debug.dump.near"      );
  auto const ffar           =  vars.getFloat           ("rssv.method.debug.dump.far"       );
  auto const fovy           =  vars.getFloat           ("rssv.method.debug.dump.fovy"      );
  auto const wireframe      =  vars.getBool            ("rssv.method.debug.wireframe"      );

  auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"                        );


  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID + gl_InstanceID;
  }

  ).";

  std::string const fsSrc = R".(
  uniform uint levelToDraw = 0;

  #if WIREFRAME == 0
  in vec3 gNormal;
  #endif

  uniform uint mode        = 0;
  const vec4 colors[] = {
    vec4(.0,.5,.0,1),
    vec4(.5,.0,.0,1),
    vec4(.0,.0,.5,1),
  };
  layout(location=0)out vec4 fColor;
  void main(){

    #if WIREFRAME == 1
    fColor = vec4(colors[mode]);
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
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

uint divRoundUp(uint x,uint y){
  return uint(x/y) + uint((x%y)>0);
}


layout(points)in;

#if WIREFRAME == 1
layout(line_strip,max_vertices=28)out;
#else
layout(triangle_strip,max_vertices=24)out;
out vec3 gNormal;
#endif

flat in uint vId[];

layout(binding=0)buffer NodePool    {uint  nodePool    [];};
layout(binding=1)buffer AABBPool    {float aabbPool    [];};
layout(binding=2)buffer TraverseData{uint  traverseData[];};

uniform mat4 view;
uniform mat4 proj;

uniform mat4 nodeView;
uniform mat4 nodeProj;

#line 122
uniform uint levelToDraw = 0;

uniform uint drawTightAABB = 0;


void main(){
  uint id   = vId[0];
  uint gId = traverseData[id];
#line 157

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

  mminX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+0];
  mmaxX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+1];
  mminY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+2];
  mmaxY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+3];
  mminZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+4];
  mmaxZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+gId*floatsPerAABB+5];

  startX = -1.f + xyz.x*levelTileSizeClipSpace[nofLevels-1].x;
  startY = -1.f + xyz.y*levelTileSizeClipSpace[nofLevels-1].y;
  endX   = min(startX + levelTileSizeClipSpace[levelToDraw].x,1.f);
  endY   = min(startY + levelTileSizeClipSpace[levelToDraw].y,1.f);
  startZ = CLUSTER_TO_Z(xyz.z                                   );
  endZ   = CLUSTER_TO_Z(xyz.z+(1u<<levelTileBits[levelToDraw].z));


#ifdef FAR_IS_INFINITE
  float e = -1.f;
  float f = -2.f * NEAR;
#else
  float e = -(FAR + NEAR) / (FAR - NEAR);
  float f = -2.f * NEAR * FAR / (FAR - NEAR);
#endif

  if(drawTightAABB != 0){
    startX = mminX;
    endX   = mmaxX;

    startY = mminY;
    endY   = mmaxY;

    startZ = DEPTH_TO_Z(mminZ);
    endZ   = DEPTH_TO_Z(mmaxZ);
  }


  mat4 M = proj*view*inverse(nodeView)*inverse(nodeProj);
#if WIREFRAME == 1
  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();
  gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();
  gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();
  gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();
#else
  mat4 N = inverse(nodeView)*inverse(nodeProj);
  gNormal = (N*vec4(0,0,-1,0)).xyz;
  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(1,0,0,0)).xyz;
  gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,0,1,0)).xyz;
  gl_Position = M*vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(-1,0,0,0)).xyz;
  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,1,0,0)).xyz;
  gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(startX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),  endY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  EndPrimitive();

  gNormal = (N*vec4(0,-1,0,0)).xyz;
  gl_Position = M*vec4(startX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-  endZ),startY*(-  endZ),e*  endZ+f,(-  endZ));EmitVertex();
  gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
  EndPrimitive();
#endif
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
      ge::gl::Shader::define("WIREFRAME",(int)wireframe),

      rssv::configShader,
      rssv::mortonShader,
      rssv::demortonShader,
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      ge::gl::Shader::define("WIREFRAME",(int)wireframe),
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawTraverseProgram",
      vs,
      gs,
      fs
      );

}

void drawTraverse(vars::Vars&vars){
  prepareDrawTraverse(vars);

  auto const cfg               = *vars.get<Config>        ("rssv.method.debug.dump.config"           );

  auto const nodeView          = *vars.get<glm::mat4>     ("rssv.method.debug.dump.viewMatrix"       );
  auto const nodeProj          = *vars.get<glm::mat4>     ("rssv.method.debug.dump.projectionMatrix" );
  auto const nodePool          =  vars.get<Buffer>        ("rssv.method.debug.dump.nodePool"         );
  auto const aabbPool          =  vars.get<Buffer>        ("rssv.method.debug.dump.aabbPool"         );

  auto const view           = *vars.get<glm::mat4>     ("rssv.method.debug.viewMatrix"           );
  auto const proj           = *vars.get<glm::mat4>     ("rssv.method.debug.projectionMatrix"     );
  auto const taToDraw       =  vars.getUint32          ("rssv.method.debug.taToDraw"             );
  auto const trToDraw       =  vars.getUint32          ("rssv.method.debug.trToDraw"             );
  auto const inToDraw       =  vars.getUint32          ("rssv.method.debug.inToDraw"             );
  auto const drawTightAABB  =  vars.getBool            ("rssv.method.debug.drawTightAABB"        );

  auto vao = vars.get<VertexArray>("rssv.method.debug.vao");

  auto prg = vars.get<Program>("rssv.method.debug.drawTraverseProgram");

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
    ;

  auto dibo = vars.get<Buffer>("rssv.method.debug.traverseData");
  dibo->bind(GL_DRAW_INDIRECT_BUFFER);
  dibo->bindBase(GL_SHADER_STORAGE_BUFFER,2);

  //std::cerr << "taBits: " << inToDraw << std::endl;
  //std::cerr << "trBits: " << inToDraw << std::endl;
  //std::cerr << "inBits: " << inToDraw << std::endl;
  uint32_t const toDraw[] = {
    taToDraw,
    trToDraw,
    inToDraw,
  };
  auto const nModes = sizeof(toDraw) / sizeof(toDraw[0]);

  for(uint32_t l=0;l<cfg.nofLevels;++l){
    for(uint32_t m=0;m<nModes;++m){
      if((toDraw[m]&(1u<<l)) != 0){
        prg->set1ui      ("mode"       ,m);
        prg->set1ui      ("levelToDraw",l);
        glDrawArraysIndirect(GL_POINTS,(GLvoid*const)(sizeof(uint32_t)*4u*(nModes*l+m)));
      }
    }
  }

  vao->unbind();

}

}
