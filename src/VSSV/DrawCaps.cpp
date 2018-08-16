#include<VSSV/DrawCaps.h>
#include<geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>

DrawCaps::DrawCaps(std::shared_ptr<Adjacency const>const&adj){
#include"VSSV/Shaders.h"
#include"SilhouetteShaders.h"
  program = std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        _drawCapsVertexShaderSrc));

  size_t const sizeofTriangleInBytes = sizeof(float)*3*3;
  caps = std::make_shared<ge::gl::Buffer>(adj->getVertices());

  vao = std::make_shared<ge::gl::VertexArray>();
  GLsizei const stride     = GLsizei(sizeofTriangleInBytes);
  GLenum  const normalized = GL_FALSE;
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const divisor    = GLuint(nofCapsPerTriangle);
  for(size_t i=0;i<3;++i){
    GLintptr offset = sizeof(float)*3 * i;
    GLuint   index = GLuint(i);
    vao->addAttrib(caps,index,3,GL_FLOAT,stride,offset,normalized,divisor);
  }

  nofTriangles = adj->getNofTriangles();

}

void DrawCaps::draw(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  program->use();
  program->setMatrix4fv("viewMatrix"      ,glm::value_ptr(viewMatrix      ));
  program->setMatrix4fv("projectionMatrix",glm::value_ptr(projectionMatrix));
  program->set4fv      ("lightPosition"   ,glm::value_ptr(lightPosition   ));
  vao->bind();
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const nofInstances = GLuint(nofCapsPerTriangle * nofTriangles);
  GLsizei const nofVertices  = GLsizei(3);
  GLint   const firstVertex  = 0;
  ge::gl::glDrawArraysInstanced(GL_TRIANGLES,firstVertex,nofVertices,nofInstances);
  vao->unbind();
}
