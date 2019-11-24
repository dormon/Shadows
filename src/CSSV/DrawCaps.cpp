#include<CSSV/DrawCaps.h>
#include<CSSV/DrawCapsProgram.h>
#include<ShadowVolumes.h>
#include<geGL/StaticCalls.h>
#include<CSSV/createCapsProgram.h>
#include<CSSV/createCapsBuffer.h>
#include<CSSV/createCapsVAO.h>

#include<Simplex.h>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace cssv;

DrawCaps::DrawCaps(Adjacency const*adj,vars::Vars&vars):vars(vars){
}

void DrawCaps::draw(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  createCapsProgram(vars);
  createCapsBuffer(vars);
  createCapsVAO(vars);

  auto adj          = vars.get<Adjacency>("adjacency");
  auto program      = vars.get<Program>("cssv.method.caps.program");
  auto vao          = vars.get<VertexArray>("cssv.method.caps.vao");

  auto nofTriangles = adj->getNofTriangles();

  auto mvp = projectionMatrix * viewMatrix;
  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp          ))
    ->set4fv      ("lightPosition",value_ptr(lightPosition));
  program->use();
  vao->bind();
  glDrawArrays(GL_TRIANGLES,0,(GLsizei)nofTriangles*3);
  vao->unbind();
}

