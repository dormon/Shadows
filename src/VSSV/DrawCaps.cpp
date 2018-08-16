#include<VSSV/DrawCaps.h>
#include<geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>

using namespace ge::gl;
using namespace std;
using namespace glm;

shared_ptr<VertexArray>createVAO(shared_ptr<Buffer>const&caps){
  auto vao = make_shared<VertexArray>();
  auto   const stride             = static_cast<GLsizei>(sizeof(Triangle3Df));
  GLenum const normalized         = GL_FALSE;
  size_t const nofCapsPerTriangle = 2;
  auto   const divisor            = static_cast<GLuint> (nofCapsPerTriangle);
  for(GLuint i=0;i<3;++i){
    GLintptr const offset = sizeof(Vertex3Df) * i;
    vao->addAttrib(caps,i,3,GL_FLOAT,stride,offset,normalized,divisor);
  }
  return vao;
}

shared_ptr<Program>createProgram(){
#include <VSSV/CapsShader.h>
#include"SilhouetteShaders.h"
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        vertexShaderSrc));
  return program;
}

DrawCaps::DrawCaps(shared_ptr<Adjacency const>const&adj){
  program      = createProgram();
  caps         = make_shared<Buffer>(adj->getVertices());
  vao          = createVAO(caps);
  nofTriangles = adj->getNofTriangles();
}

void DrawCaps::draw(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  auto const mvp = projectionMatrix * viewMatrix;
  program->setMatrix4fv("mvp"  ,value_ptr(mvp          ))
         ->set4fv      ("light",value_ptr(lightPosition))
         ->use();
  vao->bind();
  size_t  const nofCapsPerTriangle = 2;
  auto    const nofInstances = static_cast<GLuint>(nofCapsPerTriangle * nofTriangles);
  GLsizei const nofVertices  = 3;
  GLint   const firstVertex  = 0;
  glDrawArraysInstanced(GL_TRIANGLES,firstVertex,nofVertices,nofInstances);
  vao->unbind();
}
