#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>

#include <Sintorn2/debug/drawNodePool.h>

#include <Sintorn2/mortonShader.h>
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>
#include <Sintorn2/config.h>

using namespace ge::gl;
using namespace std;

namespace sintorn2::debug{

void prepareDrawNodePool(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug"
      "wavefrontSize"                        ,
      "sintorn2.method.debug.dump.windowSize",
      "sintorn2.method.debug.dump.minZBits"  ,
      "sintorn2.method.debug.dump.tileX"     ,
      "sintorn2.method.debug.dump.tileY"     ,
      "sintorn2.method.debug.dump.near"      ,
      "sintorn2.method.debug.dump.far"       ,
      "sintorn2.method.debug.dump.fovy"      ,
      );

  auto const windowSize     = *vars.get<glm::uvec2>    ("sintorn2.method.debug.dump.windowSize");
  auto const minZBits       =  vars.getUint32          ("sintorn2.method.debug.dump.minZBits"  );
  auto const tileX          =  vars.getUint32          ("sintorn2.method.debug.dump.tileX"     );
  auto const tileY          =  vars.getUint32          ("sintorn2.method.debug.dump.tileY"     );
  auto const nnear          =  vars.getFloat           ("sintorn2.method.debug.dump.near"      );
  auto const ffar           =  vars.getFloat           ("sintorn2.method.debug.dump.far"       );
  auto const fovy           =  vars.getFloat           ("sintorn2.method.debug.dump.fovy"      );

  auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"                        );


  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID;
  }

  ).";

  std::string const fsSrc = R".(
  #version 450

  layout(location=0)out vec4 fColor;
  void main(){
    fColor = vec4(1);
  }
  ).";

  std::string const gsSrc = R".(
#ifndef WARP
#define WARP 64
#endif//WARP

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

  uniform mat4 view;
  uniform mat4 proj;

  uniform mat4 nodeView;
  uniform mat4 nodeProj;

  void main(){
    const uint warpBits        = uint(ceil(log2(float(WARP))));
    const uint clustersX       = uint(WINDOW_X/TILE_X) + uint(WINDOW_X%TILE_X != 0u);
    const uint clustersY       = uint(WINDOW_Y/TILE_Y) + uint(WINDOW_Y%TILE_Y != 0u);
    const uint xBits           = uint(ceil(log2(float(clustersX))));
    const uint yBits           = uint(ceil(log2(float(clustersY))));
    const uint zBits           = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
    const uint clustersZ       = uint(1u << zBits);
    const uint allBits         = xBits + yBits + zBits;
    const uint nofLevels       = uint(allBits/warpBits) + uint(allBits%warpBits != 0u);
    const uint uintsPerWarp    = uint(WARP/32u);

    const uint warpMask        = uint(WARP - 1u);

    const uint levelSize[6] = {
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-0u)*warpBits),0)),
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-1u)*warpBits),0)),
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-2u)*warpBits),0)),
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-3u)*warpBits),0)),
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-4u)*warpBits),0)),
      uintsPerWarp << uint(max(int(allBits) - int((nofLevels-5u)*warpBits),0)),
    };

    const uint levelOffset[6] = {
      0,
      0 + levelSize[0],
      0 + levelSize[0] + levelSize[1],
      0 + levelSize[0] + levelSize[1] + levelSize[2],
      0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3],
      0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3] + levelSize[4],
    };
    uint gId = vId[0];

    uint x = gId % clustersX;
    uint y = (gId / clustersX) % clustersY;
    uint z = (gId / (clustersX * clustersY));

    uint mor = morton(uvec3(x,y,z));

    uint bit  = (mor >> (warpBits*0u)) & warpMask;
    uint node = (mor >> (warpBits*1u));

    uint doesNodeExist = nodePool[levelOffset[clamp(nofLevels-1u,0u,5u)]+node*2u+uint(bit>31u)]&(1u<<(bit&0x1fu));

    if(doesNodeExist == 0)return;

    vec4 ndc  = vec4(0);
    vec4 ndc2 = vec4(0);

    float startZ = clusterToZ(z);
    float endZ   = clusterToZ(z+1);

    float startX = -1.f + 2.f * float(x*TILE_X) / float(WINDOW_X);
    float startY = -1.f + 2.f * float(y*TILE_Y) / float(WINDOW_Y);

    float endX = clamp(-1.f + 2.f * float((x+1)*TILE_X) / float(WINDOW_X),-1.f,1.f);
    float endY = clamp(-1.f + 2.f * float((y+1)*TILE_Y) / float(WINDOW_Y),-1.f,1.f);



#ifdef FAR_IS_INFINITE
    float e = -1.f;
    float f = -2.f * NEAR;
#else
    float e = -(FAR + NEAR) / (FAR - NEAR);
    float f = -2.f * NEAR * FAR / (FAR - NEAR);
#endif

    mat4 M = proj*view*inverse(nodeView)*inverse(nodeProj);
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

    /*
    M = proj*view;//*inverse(nodeView)*inverse(nodeProj);
    startZ = -1.f;
    gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
    gl_Position = M*vec4(  endX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
    gl_Position = M*vec4(  endX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
    gl_Position = M*vec4(startX*(-startZ),  endY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
    gl_Position = M*vec4(startX*(-startZ),startY*(-startZ),e*startZ+f,(-startZ));EmitVertex();
    EndPrimitive();
    */

    /*
    gl_Position = vec4(-1,0,0,1);EmitVertex();
    gl_Position = vec4(+1,0,0,1);EmitVertex();
    EndPrimitive();
    */

    //*
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
    // */
  }

  ).";

  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      ge::gl::Shader::define("WARP"      ,(uint32_t)wavefrontSize),
      ge::gl::Shader::define("WINDOW_X"  ,(uint32_t)windowSize.x ),
      ge::gl::Shader::define("WINDOW_Y"  ,(uint32_t)windowSize.y ),
      ge::gl::Shader::define("MIN_Z_BITS",(uint32_t)minZBits     ),
      ge::gl::Shader::define("NEAR"      ,nnear                  ),
      glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
      ge::gl::Shader::define("FOVY"      ,fovy                   ),
      ge::gl::Shader::define("TILE_X"    ,tileX                  ),
      ge::gl::Shader::define("TILE_Y"    ,tileY                  ),

      sintorn2::mortonShader,
      sintorn2::depthToZShader,
      sintorn2::quantizeZShader,
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,fsSrc);

  vars.reCreate<Program>(
      "sintorn2.method.debug.drawNodePoolProgram",
      vs,
      gs,
      fs
      );

}

void drawNodePool(vars::Vars&vars){
  prepareDrawNodePool(vars);


  auto const windowSize     = *vars.get<glm::uvec2>    ("sintorn2.method.debug.dump.windowSize"      );
  auto const minZBits       =  vars.getUint32          ("sintorn2.method.debug.dump.minZBits"        );
  auto const tileX          =  vars.getUint32          ("sintorn2.method.debug.dump.tileX"           );
  auto const tileY          =  vars.getUint32          ("sintorn2.method.debug.dump.tileY"           );
  auto const nnear          =  vars.getFloat           ("sintorn2.method.debug.dump.near"            );
  auto const ffar           =  vars.getFloat           ("sintorn2.method.debug.dump.far"             );
  auto const fovy           =  vars.getFloat           ("sintorn2.method.debug.dump.fovy"            );
  auto const nodeView       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.viewMatrix"      );
  auto const nodeProj       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.projectionMatrix");
  auto const nodePool       =  vars.get<Buffer>        ("sintorn2.method.debug.dump.nodePool"        );
  auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"                              );

  auto const view           = *vars.get<glm::mat4>     ("sintorn2.method.debug.viewMatrix"           );
  auto const proj           = *vars.get<glm::mat4>     ("sintorn2.method.debug.projectionMatrix"     );

  auto vao = vars.get<VertexArray>("sintorn2.method.debug.vao");

  auto cfg = Config(wavefrontSize,windowSize.x,windowSize.y,tileX,tileY,minZBits,nnear,ffar,fovy);

  auto prg = vars.get<Program>("sintorn2.method.debug.drawNodePoolProgram");

  vao->bind();
  nodePool->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  prg->use();
  prg
    ->setMatrix4fv("nodeView",glm::value_ptr(nodeView))
    ->setMatrix4fv("nodeProj",glm::value_ptr(nodeProj))
    ->setMatrix4fv("view"    ,glm::value_ptr(view    ))
    ->setMatrix4fv("proj"    ,glm::value_ptr(proj    ))
    ;
  glDrawArrays(GL_POINTS,0,cfg.clustersX*cfg.clustersY*cfg.clustersZ);

  vao->unbind();

}

}
