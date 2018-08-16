#include<VSSV/DrawCaps.h>
#include<geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>

using namespace ge::gl;
using namespace std;

shared_ptr<VertexArray>createVAO(shared_ptr<Buffer>const&caps){
  auto vao = make_shared<VertexArray>();
  GLsizei const stride     = GLsizei(sizeof(Triangle3Df));
  GLenum  const normalized = GL_FALSE;
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const divisor    = GLuint(nofCapsPerTriangle);
  for(size_t i=0;i<3;++i){
    GLintptr offset = sizeof(Vertex3Df) * i;
    GLuint   index = GLuint(i);
    vao->addAttrib(caps,index,3,GL_FLOAT,stride,offset,normalized,divisor);
  }
  return vao;
}

shared_ptr<Program>createProgram(){
#include"VSSV/Shaders.h"
#include"SilhouetteShaders.h"
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        _drawCapsVertexShaderSrc));
  return program;
}

DrawCaps::DrawCaps(shared_ptr<Adjacency const>const&adj){
  program      = createProgram();
  caps         = make_shared<Buffer>(adj->getVertices());
  vao          = createVAO(caps);
  nofTriangles = adj->getNofTriangles();
}

void DrawCaps::draw(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  program->setMatrix4fv("viewMatrix"      ,glm::value_ptr(viewMatrix      ))
         ->setMatrix4fv("projectionMatrix",glm::value_ptr(projectionMatrix))
         ->set4fv      ("lightPosition"   ,glm::value_ptr(lightPosition   ))
         ->use();
  vao->bind();
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const nofInstances = GLuint(nofCapsPerTriangle * nofTriangles);
  GLsizei const nofVertices  = GLsizei(3);
  GLint   const firstVertex  = 0;
  glDrawArraysInstanced(GL_TRIANGLES,firstVertex,nofVertices,nofInstances);
  vao->unbind();
}
