#include<glm/gtc/type_ptr.hpp>

#include<Vars/Vars.h>
#include<geGL/StaticCalls.h>
#include<geGL/geGL.h>

#include<FastAdjacency.h>

#include<CSSV/caps/createProgram.h>
#include<CSSV/caps/createBuffer.h>
#include<CSSV/caps/createVAO.h>
#include<CSSV/caps/draw.h>

using namespace std;
using namespace ge::gl;
using namespace glm;

void cssv::drawCaps(
    vars::Vars&vars,
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

