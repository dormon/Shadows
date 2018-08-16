#include<VSSV/DrawCaps.h>
#include<geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>

using namespace ge::gl;

DrawCaps::DrawCaps(std::shared_ptr<Adjacency const>const&adj){
#include"VSSV/Shaders.h"
#include"SilhouetteShaders.h"
  program = std::make_shared<Program>(
      std::make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        _drawCapsVertexShaderSrc));

  caps = std::make_shared<Buffer>(adj->getVertices());

  vao = std::make_shared<VertexArray>();
  GLsizei const stride     = GLsizei(sizeof(Triangle3Df));
  GLenum  const normalized = GL_FALSE;
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const divisor    = GLuint(nofCapsPerTriangle);
  for(size_t i=0;i<3;++i){
    GLintptr offset = sizeof(Vertex3Df) * i;
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
  glDrawArraysInstanced(GL_TRIANGLES,firstVertex,nofVertices,nofInstances);
  vao->unbind();
}
