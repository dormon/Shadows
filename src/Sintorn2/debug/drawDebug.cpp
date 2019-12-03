#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <imguiDormon/imgui.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>

#include <Sintorn2/mortonShader.h>
#include <Sintorn2/debug/drawDebug.h>
#include <Sintorn2/debug/dumpData.h>
#include <Sintorn2/debug/drawSamples.h>
#include <Sintorn2/debug/drawNodePool.h>
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>

using namespace ge::gl;
using namespace std;

namespace sintorn2::debug{
void prepareCommon(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug");
  vars.reCreate<VertexArray>("sintorn2.method.debug.vao");
}


void blitDepth(vars::Vars&vars){
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  glBlitNamedFramebuffer(
      gBuffer->fbo->getId(),
      0,
      0,0,windowSize.x,windowSize.y,
      0,0,windowSize.x,windowSize.y,
      GL_DEPTH_BUFFER_BIT,
      GL_NEAREST);
}

void createDebugProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug");

  std::string const vs = 
R".(
#version 450

uniform mat4 view = mat4(1);
uniform mat4 proj = mat4(1);

void main(){
  if(gl_VertexID == 0)gl_Position = proj*view*vec4(0,0,0,1);
  if(gl_VertexID == 1)gl_Position = proj*view*vec4(10,0,0,1);
  if(gl_VertexID == 2)gl_Position = proj*view*vec4(0,10,0,1);
}
).";
  std::string const fs = 
R".(
#version 450

layout(location=0)out vec4 fColor;

void main(){
  fColor = vec4(1,0,0,0);
}
).";

  vars.reCreate<Program>("sintorn2.method.debug.basicProgram",
      make_shared<Shader>(GL_VERTEX_SHADER,vs),
      make_shared<Shader>(GL_FRAGMENT_SHADER,fs));


}

void prepareDrawMortons(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug"
      "windowSize",
      "wavefrontSize",
      "args.camera.near",
      "args.camera.far",
      "args.camera.fovy",
      "sintorn2.param.minZBits",
      "sintorn2.param.tileX"   ,
      "sintorn2.param.tileY"   ,
      );
  std::string const vs = 
R".(
#version 450
void main(){
  gl_Position = vec4(-1+2*(gl_VertexID&1),-1+2*(gl_VertexID>>1),-1,1);
}
).";

  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"          );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"             );
  auto const nnear               =  vars.getFloat           ("args.camera.near"       );
  auto const ffar                =  vars.getFloat           ("args.camera.far"        );
  auto const fovy                =  vars.getFloat           ("args.camera.fovy"       );
  auto const minZBits            =  vars.getUint32          ("sintorn2.param.minZBits");
  auto const tileX               =  vars.getUint32          ("sintorn2.param.tileX"   );
  auto const tileY               =  vars.getUint32          ("sintorn2.param.tileY"   );

  std::string const fs = 
R".(

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

layout(location=0)out vec4 fColor;
layout(binding=1)uniform sampler2DRect depthTexture;

uint getMorton(uvec2 coord){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}

vec3 hue(float t){
  t = fract(t);
  if(t<1/6.)return mix(vec3(1,0,0),vec3(1,1,0),(t-0/6.)*6);
  if(t<2/6.)return mix(vec3(1,1,0),vec3(0,1,0),(t-1/6.)*6);
  if(t<3/6.)return mix(vec3(0,1,0),vec3(0,1,1),(t-2/6.)*6);
  if(t<4/6.)return mix(vec3(0,1,1),vec3(0,0,1),(t-3/6.)*6);
  if(t<5/6.)return mix(vec3(0,0,1),vec3(1,0,1),(t-4/6.)*6);
            return mix(vec3(1,0,1),vec3(1,0,0),(t-5/6.)*6);
}

float scramble(uint x){
  x = x*(x*(x*(x*(x+1)+1)+1)+1)+1;
  return float(x) / float(0xffffffff);
}

void main(){
  uint morton = getMorton(uvec2(gl_FragCoord.xy));
  
  vec3 color = hue(float(morton) * 3.1415926534932f);
  fColor = vec4(color,0);
}
).";

  vars.reCreate<Program>("sintorn2.method.debug.drawMortonsProgram",
      make_shared<Shader>(GL_VERTEX_SHADER,vs),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
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
        fs));

}

void drawBasic(vars::Vars&vars){
  debug::createDebugProgram(vars);
  debug::blitDepth(vars);
  glEnable(GL_DEPTH_TEST);
  //ge::gl::glViewport(0,0,100,100);
  //ge::gl::glClear(GL_COLOR_BUFFER_BIT);
  //ge::gl::glClearColor(1,0,0,1);
  //ge::gl::glViewport(0,0,512,512);

  auto prg = vars.get<Program>("sintorn2.method.debug.basicProgram");
  auto vao = vars.get<VertexArray>("sintorn2.method.debug.vao");
  auto view = *vars.get<glm::mat4>("sintorn2.method.debug.viewMatrix");
  auto proj = *vars.get<glm::mat4>("sintorn2.method.debug.projectionMatrix");

  vao->bind();
  prg->use();
  prg
    ->setMatrix4fv("view",glm::value_ptr(view))
    ->setMatrix4fv("proj",glm::value_ptr(proj));
  glDrawArrays(GL_TRIANGLES,0,3);
  vao->unbind();
}

void drawMortons(vars::Vars&vars){
  prepareDrawMortons(vars);

  auto depth       = vars.get<GBuffer>("gBuffer")->depth;
  depth->bind(1);
  auto prg = vars.get<Program>("sintorn2.method.debug.drawMortonsProgram");
  prg->use();
  

  auto vao = vars.get<VertexArray>("sintorn2.method.debug.vao");
  vao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  vao->unbind();
}


}

void sintorn2::drawDebug(vars::Vars&vars){
  debug::prepareCommon(vars);


  enum DebugType{
    DEFAULT,
    DRAW_MORTON,
    DRAW_SAMPLES,
    DRAW_NODEPOOL,
  };

  auto&type         = vars.addOrGetUint32("sintorn2.method.debug.type",DEFAULT);
  auto&levelsToDraw = vars.addOrGetUint32("sintorn2.method.debug.levelsToDraw",0);

  if(ImGui::BeginMainMenuBar()){
    if(ImGui::BeginMenu("debug")){
      if(ImGui::MenuItem("default"))
        type = DEFAULT;
      if(ImGui::MenuItem("drawMorton"))
        type = DRAW_MORTON;
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("dump")){
      if(ImGui::MenuItem("copyData"))
        sintorn2::debug::dumpData(vars);
      if(ImGui::MenuItem("drawSamples"))
        type = DRAW_SAMPLES;
      if(ImGui::MenuItem("drawNodePool"))
        type = DRAW_NODEPOOL;

      if(type == DRAW_NODEPOOL){
        if(vars.has("sintorn2.method.debug.dump.nofLevels")){
          auto nofLevels = vars.getUint32("sintorn2.method.debug.dump.nofLevels");
          for(uint32_t i=0;i<nofLevels;++i){
            std::stringstream ss;
            ss << "level" << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              levelsToDraw ^= 1<<i;
            }
          }
        }
      }

      ImGui::EndMenu();
    }


    ImGui::EndMainMenuBar();
  }
  if(type == DRAW_MORTON)
    debug::drawMortons(vars);

  //if(type == BASIC)
  //  debug::drawBasic(vars);

  if(type == DRAW_SAMPLES)
    debug::drawSamples(vars);

  if(type == DRAW_NODEPOOL){
    debug::drawSamples(vars);
    debug::drawNodePool(vars);
  }


  //if(ImGui::Button("mine button"))
  //  std::cerr << "button pressed" << std::endl;
  /*
  */
}
