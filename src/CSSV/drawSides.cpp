#include <geGL/geGL.h>
#include <Vars/Vars.h>
#include <CSSV/drawSides.h>
#include <CSSV/createDrawSidesProgram.h>
#include <CSSV/createDrawSidesVAO.h>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/StaticCalls.h>

using namespace glm;
using namespace ge::gl;
using namespace std;

void cssv::drawSides(
    vars::Vars&vars,
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  createDrawSidesVAO(vars);
  cssv::createDrawSidesProgram(vars);

  auto dibo    = vars.get<Buffer     >("cssv.method.dibo"             );
  auto vao     = vars.get<VertexArray>("cssv.method.drawSides.vao"    );
  auto program = vars.get<Program    >("cssv.method.drawSides.program");

  auto mvp = projectionMatrix * viewMatrix;
  dibo->bind(GL_DRAW_INDIRECT_BUFFER);
  vao->bind();
  program->use();
  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp          ))
    ->set4fv      ("lightPosition",value_ptr(lightPosition));
  glPatchParameteri(GL_PATCH_VERTICES,2);
  glDrawArraysIndirect(GL_PATCHES,NULL);
  vao->unbind();
}
