#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>

#include <Sintorn2/drawDebug.h>

using namespace ge::gl;
using namespace std;

namespace sintorn2{
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

  vars.reCreate<Program>("sintorn2.method.debug.program",
      make_shared<Shader>(GL_VERTEX_SHADER,vs),
      make_shared<Shader>(GL_FRAGMENT_SHADER,fs));

  vars.reCreate<VertexArray>("sintorn2.method.debug.vao");

  auto blit = vars.reCreate<Framebuffer>("sintorn2.method.debug.blitFbo");
}
}

void sintorn2::drawDebug(vars::Vars&vars){
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  glBlitNamedFramebuffer(
      gBuffer->fbo->getId(),
      0,
      0,0,windowSize.x,windowSize.y,
      0,0,windowSize.x,windowSize.y,
      GL_DEPTH_BUFFER_BIT,
      GL_NEAREST);


  createDebugProgram(vars);
  glEnable(GL_DEPTH_TEST);
  //ge::gl::glViewport(0,0,100,100);
  //ge::gl::glClear(GL_COLOR_BUFFER_BIT);
  //ge::gl::glClearColor(1,0,0,1);
  //ge::gl::glViewport(0,0,512,512);

  auto prg = vars.get<Program>("sintorn2.method.debug.program");
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
