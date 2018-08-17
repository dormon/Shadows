#include <VSSV/DrawSides.h>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/StaticCalls.h>

using namespace ge::gl;
using namespace glm;
using namespace std;

DrawSides::DrawSides(vars::Vars&vars):vars(vars){}

void DrawSides::draw(
  glm::vec4 const&light     ,
  glm::mat4 const&view      ,
  glm::mat4 const&projection){
  auto const mvp = projection * view;
  program->setMatrix4fv("mvp"  ,value_ptr(mvp  ))
         ->set4fv      ("light",value_ptr(light))
         ->use();
  vao->bind();
  if(vars.getBool("vssv.useStrips"))
    glDrawArraysInstanced(GL_TRIANGLE_STRIP,0,4,GLsizei(nofEdges*maxMultiplicity));
  else
    glDrawArraysInstanced(GL_TRIANGLES     ,0,6,GLsizei(nofEdges*maxMultiplicity));
  vao->unbind();
}
