#include<CSSV/DrawSides.h>
#include<ShadowMethod.h>
#include<geGL/StaticCalls.h>
#include<CSSV/DrawSidesProgram.h>

using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;

shared_ptr<VertexArray>createSidesVao(Buffer*silhouettes){
  auto vao = make_shared<VertexArray>();
  vao->addAttrib(silhouettes,0,componentsPerVertex4D,GL_FLOAT);
  return vao;
}

DrawSides::DrawSides(Buffer*silhouettes,Buffer*d):dibo(d){
  vao     = createSidesVao(silhouettes);
  program = createDrawSidesProgram();
}

void DrawSides::draw(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
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
