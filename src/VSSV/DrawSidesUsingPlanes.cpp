#include <VSSV/DrawSidesUsingPlanes.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>
#include <ShadowMethod.h>

using namespace std;
using namespace ge::gl;
using namespace glm;

namespace vssvUsingPlanes{

template<size_t N>
struct GPUEdgeData{
  Vertex3Df vertexA;
  Vertex3Df vertexB;
  float     nofOpposite;
  Vertex4Df planes[N];
};

void writePlane(Vertex4Df&plane,Vertex3Df const*const src,size_t e,size_t o,shared_ptr<Adjacency const>const&adj){
  auto const glmPlane = computePlane(
          toVec3(src[adj->getEdge(e,0)/3].elements    ),
          toVec3(src[adj->getEdge(e,1)/3].elements    ),
          toVec3(src[adj->getOpposite(e,o)/3].elements));
  for(int i=0;i<4;++i)
    plane.elements[i] = glmPlane[i];
}

void writePlanes(Vertex4Df *const planes,Vertex3Df const*const src,size_t e,shared_ptr<Adjacency const>const&adj){
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    writePlane(planes[o],src,e,o,adj);
}

template<size_t N>
void writeEdge(GPUEdgeData<N>&edge,Vertex3Df const*const src,size_t e,shared_ptr<Adjacency const>const&adj){
  edge.vertexA = src[adj->getEdge(e,0)/3];
  edge.vertexB = src[adj->getEdge(e,1)/3];
  edge.nofOpposite = adj->getNofOpposite(e);

  writePlanes(edge.planes,src,e,adj);
}

template<size_t N>
void writeEdges(vector<GPUEdgeData<N>>&dst,Vertex3Df const*const src,shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst[e],src,e,adj);
}

template<size_t N>
shared_ptr<Buffer>createSidesBuffer(shared_ptr<Adjacency const>const&adj){
  auto const src = reinterpret_cast<Vertex3Df const*>(adj->getVertices().data());
  vector<GPUEdgeData<N>>dst(adj->getNofEdges());
  writeEdges(dst,src,adj);
  return make_shared<Buffer>(dst);
}

template<size_t N>
shared_ptr<VertexArray>createVAO(shared_ptr<Buffer>const&sides){
  auto vao = make_shared<VertexArray>();
  GLenum const normalized = GL_FALSE;
  GLuint const divisor = N;
  GLintptr offset = 0;
  GLsizei const stride = static_cast<GLsizei>(sizeof(GPUEdgeData<N>));
  vao->addAttrib(sides,0,3,GL_FLOAT,stride,offset,normalized,divisor);
  offset += 3*sizeof(float);
  vao->addAttrib(sides,1,3,GL_FLOAT,stride,offset,normalized,divisor);
  offset += 3*sizeof(float);
  vao->addAttrib(sides,2,1,GL_FLOAT,stride,offset,normalized,divisor);
  offset += sizeof(float);
  for(GLuint o=0;o<N;++o){
    vao->addAttrib(sides,3+o,4,GL_FLOAT,stride,offset,normalized,divisor);
    offset += 4*sizeof(float);
  }
  return vao;
}

shared_ptr<Program>createProgram(vars::Vars const&vars){
#include<VSSV/DrawSidesUsingPlanesShader.h>
#include<SilhouetteShaders.h>

  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        vars.getBool("vssv.usePlanes"             )?Shader::define("USE_PLANES"               ):"",
        vars.getBool("vssv.useStrips"             )?Shader::define("USE_TRIANGLE_STRIPS"      ):"",
        vars.getBool("vssv.useAllOppositeVertices")?Shader::define("USE_ALL_OPPOSITE_VERTICES"):"",
        silhouetteFunctions,
        vertexShaderSrc));
  return program;
}

}

DrawSidesUsingPlanes::DrawSidesUsingPlanes(vars::Vars&vars,shared_ptr<Adjacency const>const&adj):vars(vars){
  sides    = vssvUsingPlanes::createSidesBuffer<2>(adj);
  vao      = vssvUsingPlanes::createVAO<2>(sides);
  program  = vssvUsingPlanes::createProgram(vars);
  nofEdges = adj->getNofEdges();
  maxMultiplicity = 2;
}

void DrawSidesUsingPlanes::draw(
    vec4 const&light     ,
    mat4 const&view      ,
    mat4 const&projection){
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
