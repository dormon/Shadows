#include <VSSV/DrawCaps.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>
#include <Vars/Vars.h>

using namespace ge::gl;
using namespace std;
using namespace glm;

shared_ptr<VertexArray>createVAO(shared_ptr<Buffer>const&caps){
  auto vao = make_shared<VertexArray>();
  auto   const stride             = static_cast<GLsizei>(sizeof(Triangle3Df));
  GLuint const nofCapsPerTriangle = 2;
  for(GLuint i=0;i<3;++i){
    GLintptr const offset = sizeof(Vertex3Df) * i;
    vao->addAttrib(caps,i,3,GL_FLOAT,stride,offset,GL_FALSE,nofCapsPerTriangle);
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
  auto const nofCaps = static_cast<GLuint>(2 * nofTriangles);
  glDrawArraysInstanced(GL_TRIANGLES,0,3,nofCaps);
  vao->unbind();
}
