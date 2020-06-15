#include <RSSV/debug/drawStencil.h>
#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <RSSV/config.h>

#include <Deferred.h>
#include <FunctionPrologue.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawStencil(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      "rssv.method.config"    
      );


  std::string const vsSrc = R".(
  #version 450

  void main(){
    gl_Position = vec4(gl_VertexID&1,gl_VertexID>>1,0,1)*2-1;
  }

  ).";

  std::string const fsSrc = R".(

  const vec4 colors[6] = {
    vec4(.1,.1,.1,1)*5*2,
    vec4(.1,.0,.0,1)*5*2,
    vec4(.0,.1,.0,1)*5*2,
    vec4(.0,.0,.1,1)*5*2,
    vec4(.1,.1,.0,1)*5*2,
    vec4(.1,.0,.1,1)*5*2,
  };
  layout(location=0)out vec4 fColor;
  layout(r32i,binding=2)          uniform iimage2D      stencil     ;
  void main(){
    int value = imageLoad(stencil,ivec2(gl_FragCoord.xy)).r;

    vec3 endColor;
    if(value ==  1)endColor = vec3(.5,0,0);
    if(value ==  2)endColor = vec3(0,1,0);
    if(value ==  3)endColor = vec3(1,1,0);
    if(value == -1)endColor = vec3(0,1,1);
    if(value == -2)endColor = vec3(0,0,1);
    if(value == -3)endColor = vec3(0,0,.5);


    fColor = vec4(endColor,1);
    if(value == 0)discard;
  }
  ).";


  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawStencilProgram",
      vs,
      fs
      );

}


void drawStencil(vars::Vars&vars){
  prepareDrawStencil(vars);
  auto vao = vars.get<VertexArray>("rssv.method.debug.vao");

  auto prg = vars.get<Program>("rssv.method.debug.drawStencilProgram");
  auto stencil = vars.get<Texture>("rssv.method.stencil"                   );
  stencil->bindImage(2);

  vao->bind();
  prg->use();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);

  vao->unbind();

}

}
