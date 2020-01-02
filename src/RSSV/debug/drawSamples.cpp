#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>

#include <RSSV/debug/drawSamples.h>
#include <RSSV/config.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void createDrawSamplesProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug");

  std::string const vs = 
R".(
#version 450

uniform mat4 view = mat4(1);
uniform mat4 proj = mat4(1);

layout(binding=0)buffer Samples{float samples[];};

out vec3 vColor;
out vec3 vNormal;

void main(){

  vec3 position;
  vec3 normal;
  vec3 color;
  position[0] = samples[gl_VertexID*9+0+0];
  position[1] = samples[gl_VertexID*9+0+1];
  position[2] = samples[gl_VertexID*9+0+2];
  normal  [0] = samples[gl_VertexID*9+3+0];
  normal  [1] = samples[gl_VertexID*9+3+1];
  normal  [2] = samples[gl_VertexID*9+3+2];
  color   [0] = samples[gl_VertexID*9+6+0];
  color   [1] = samples[gl_VertexID*9+6+1];
  color   [2] = samples[gl_VertexID*9+6+2];

  vColor = color;
  vNormal = normal;
  gl_Position = proj*view*vec4(position,1);
}
).";
  std::string const fs = 
R".(
#version 450

in vec3 vColor;
in vec3 vNormal;

layout(location=0)out vec4 fColor;

void main(){
  fColor = vec4(vColor,0);
}
).";

  vars.reCreate<Program>("rssv.method.debug.drawSamplesProgram",
      make_shared<Shader>(GL_VERTEX_SHADER,vs),
      make_shared<Shader>(GL_FRAGMENT_SHADER,fs));

}

void drawSamples(vars::Vars&vars){
  FUNCTION_CALLER();
  createDrawSamplesProgram(vars);

  glEnable(GL_DEPTH_TEST);
  ge::gl::glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  auto prg  =  vars.get<Program    >("rssv.method.debug.drawSamplesProgram");
  auto vao  =  vars.get<VertexArray>("rssv.method.debug.vao");
  auto view = *vars.get<glm::mat4  >("rssv.method.debug.viewMatrix");
  auto proj = *vars.get<glm::mat4  >("rssv.method.debug.projectionMatrix");
  auto cfg  = *vars.get<Config     >("rssv.method.debug.dump.config");

  auto buf = vars.get<Buffer>("rssv.method.debug.dump.samples");
  buf->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  vao->bind();
  prg->use();
  prg
    ->setMatrix4fv("view",glm::value_ptr(view))
    ->setMatrix4fv("proj",glm::value_ptr(proj));
 
  
  glDrawArrays(GL_POINTS,0,cfg.windowX*cfg.windowY);
  vao->unbind();
}

}
