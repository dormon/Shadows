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

#include <Sintorn2/debug/drawNodePool.h>

#include <Sintorn2/mortonShader.h>
#include <Sintorn2/configShader.h>
#include <Sintorn2/config.h>


using namespace ge::gl;
using namespace std;

namespace sintorn2::debug{

void prepareDrawNodePool(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug"
      "wavefrontSize"                        ,
      "sintorn2.method.debug.dump.config"    ,
      "sintorn2.method.debug.dump.near"      ,
      "sintorn2.method.debug.dump.far"       ,
      "sintorn2.method.debug.dump.fovy"      ,
      "sintorn2.method.debug.wireframe"      ,
      );

  auto const cfg            = *vars.get<Config>        ("sintorn2.method.debug.dump.config"    );
  auto const nnear          =  vars.getFloat           ("sintorn2.method.debug.dump.near"      );
  auto const ffar           =  vars.getFloat           ("sintorn2.method.debug.dump.far"       );
  auto const fovy           =  vars.getFloat           ("sintorn2.method.debug.dump.fovy"      );
  auto const wireframe      =  vars.getBool            ("sintorn2.method.debug.wireframe"      );

  auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"                        );


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

layout(binding=0,std430)buffer NodePool   {uint  nodePool   [];};
layout(binding=1,std430)buffer AABBPool   {float aabbPool   [];};
layout(binding=2,std430)buffer AABBPointer{uint  aabbPointer[];};

uniform mat4 view;
uniform mat4 proj;

uniform mat4 nodeView;
uniform mat4 nodeProj;

uniform int memoryOptim = 0;

#line 122
uniform uint levelToDraw = 0;

uniform uint drawTightAABB = 0;

uniform uint usePrecomputedSize = 0;

/*
void getClipAABB(inout vec3 minCorner,inout vec3 maxCorner,uint node,uint level){
  uvec3 coord = demorton(node << (warpBits*(nofLevels-1-level)));
  minCorner.xy = -1.f + 2.f*vec2(coord.xy << uvec2(tileBitsX,tileBitsY)) / vec2(WINDOW_X,WINDOW_Y);
  minCorner.z  = clusterToZ(coord.z);

  const uint xBitsAvail[] = {
    0,

  };
  
  
}
*/

void main(){
  uint gId = vId[0];
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


 

  if(usePrecomputedSize == 1u){
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

  }else{

    uint bitsToDiv = warpBits*(nofLevels-1-levelToDraw);
    uint xBitsToDiv = divRoundUp(bitsToDiv , 3u);
    uint yBitsToDiv = divRoundUp(uint(max(int(bitsToDiv)-1,0)) , 3u);
    uint zBitsToDiv = divRoundUp(uint(max(int(bitsToDiv)-2,0)) , 3u);

    uint clusX = divRoundUp(clustersX,1u<<xBitsToDiv);
    uint clusY = divRoundUp(clustersY,1u<<yBitsToDiv);

    uint x = gId % clusX;
    uint y = (gId / clusX) % clusY;
    uint z = (gId / (clusX * clusY));

    uint mor = morton(uvec3(x<<xBitsToDiv,y<<yBitsToDiv,z<<zBitsToDiv));
    uint bit  = (mor >> (warpBits*(nofLevels-1-levelToDraw))) & warpMask;
    uint node = (mor >> (warpBits*(nofLevels  -levelToDraw)));


    uint doesNodeExist = nodePool[nodeLevelOffsetInUints[clamp(levelToDraw,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu));

    if(doesNodeExist == 0)return;

    uint aabbNode = (mor >> (warpBits*(nofLevels-1-levelToDraw)));

    if(memoryOptim == 1){
      uint w = aabbPointer[nodeLevelOffset[clamp(levelToDraw,0u,5u)]+aabbNode+1];
      mminX = aabbPool[w*floatsPerAABB+0];
      mmaxX = aabbPool[w*floatsPerAABB+1];
      mminY = aabbPool[w*floatsPerAABB+2];
      mmaxY = aabbPool[w*floatsPerAABB+3];
      mminZ = aabbPool[w*floatsPerAABB+4];
      mmaxZ = aabbPool[w*floatsPerAABB+5];
    }else{
      mminX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+0];
      mmaxX = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+1];
      mminY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+2];
      mmaxY = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+3];
      mminZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+4];
      mmaxZ = aabbPool[aabbLevelOffsetInFloats[clamp(levelToDraw,0u,5u)]+aabbNode*floatsPerAABB+5];
    }

    startZ = CLUSTER_TO_Z((z  )<<levelTileBits[levelToDraw].z);
    endZ   = CLUSTER_TO_Z((z+1)<<levelTileBits[levelToDraw].z);
    startX = -1.f + x*levelTileSizeClipSpace[levelToDraw].x;
    startY = -1.f + y*levelTileSizeClipSpace[levelToDraw].y;
    endX   = clamp(-1.f + (x+1)*levelTileSizeClipSpace[levelToDraw].x,-1.f,1.f);
    endY   = clamp(-1.f + (y+1)*levelTileSizeClipSpace[levelToDraw].y,-1.f,1.f);
  }


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

      sintorn2::configShader,
      sintorn2::mortonShader,
      sintorn2::demortonShader,
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      ge::gl::Shader::define("WIREFRAME",(int)wireframe),
      fsSrc);

  vars.reCreate<Program>(
      "sintorn2.method.debug.drawNodePoolProgram",
      vs,
      gs,
      fs
      );

}

void drawNodePool(vars::Vars&vars){
  prepareDrawNodePool(vars);

  auto const cfg            = *vars.get<Config>        ("sintorn2.method.debug.dump.config"          );

  auto const nodeView       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.viewMatrix"      );
  auto const nodeProj       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.projectionMatrix");
  auto const nodePool       =  vars.get<Buffer>        ("sintorn2.method.debug.dump.nodePool"        );
  auto const aabbPool       =  vars.get<Buffer>        ("sintorn2.method.debug.dump.aabbPool"        );

  auto const view           = *vars.get<glm::mat4>     ("sintorn2.method.debug.viewMatrix"           );
  auto const proj           = *vars.get<glm::mat4>     ("sintorn2.method.debug.projectionMatrix"     );
  auto const levelsToDraw   =  vars.getUint32          ("sintorn2.method.debug.levelsToDraw"         );
  auto const drawTightAABB  =  vars.getBool            ("sintorn2.method.debug.drawTightAABB"        );
  auto const memoryOptim    =  vars.getInt32           ("sintorn2.method.debug.dump.memoryOptim"     );

  auto vao = vars.get<VertexArray>("sintorn2.method.debug.vao");

  auto prg = vars.get<Program>("sintorn2.method.debug.drawNodePoolProgram");
  auto usePrecomputedSize = vars.getBool("sintorn2.method.debug.usePrecomputedSize");

  vao->bind();
  nodePool->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  prg->use();

  if(memoryOptim){
    auto aabbPointer = vars.get<Buffer>("sintorn2.method.debug.dump.aabbPointer");
    aabbPointer->bindBase(GL_SHADER_STORAGE_BUFFER,2);
    prg->set1i("memoryOptim",memoryOptim);
  }

  prg
    ->setMatrix4fv("nodeView"   ,glm::value_ptr(nodeView))
    ->setMatrix4fv("nodeProj"   ,glm::value_ptr(nodeProj))
    ->setMatrix4fv("view"       ,glm::value_ptr(view    ))
    ->setMatrix4fv("proj"       ,glm::value_ptr(proj    ))
    ->set1ui      ("drawTightAABB",(uint32_t)drawTightAABB)
    ->set1ui      ("usePrecomputedSize",(uint32_t)usePrecomputedSize)
    ;


  for(uint32_t l=0;l<cfg.nofLevels;++l){
    if((levelsToDraw&(1u<<l)) == 0)continue;
    prg->set1ui      ("levelToDraw",l);
    glDrawArrays(GL_POINTS,0,cfg.nofNodesPerLevel[l]);
  }

  vao->unbind();

}

}
