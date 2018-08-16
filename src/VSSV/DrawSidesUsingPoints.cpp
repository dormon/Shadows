#include <VSSV/DrawSidesUsingPoints.h>
#include <glm/gtc/type_ptr.hpp>
#include <Simplex.h>
#include <FastAdjacency.h>
#include <geGL/StaticCalls.h>

using namespace ge::gl;
using namespace glm;
using namespace std;

template<size_t N>
struct GPUEdgeData{
  Vertex3Df vertexA;
  Vertex3Df vertexB;
  float     nofOpposite;
  Vertex3Df oppositeVertices[N];
};

template<size_t N>
void writeEdge(GPUEdgeData<N>&edge,Vertex3Df const*const vertices,size_t e,shared_ptr<Adjacency const>const&adj){
  edge.vertexA     = vertices[adj->getEdgeVertexA(e)/3];
  edge.vertexB     = vertices[adj->getEdgeVertexB(e)/3];
  edge.nofOpposite = adj->getNofOpposite(e);
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    edge.oppositeVertices[o] = vertices[adj->getOpposite(e,o)/3];
  for(size_t o=adj->getNofOpposite(e);o<N;++o)
    edge.oppositeVertices[o].clear();
}

template<size_t N>
void writeEdges(vector<GPUEdgeData<N>>&dst,Vertex3Df const*const src,shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst[e],src,e,adj);
}

template<size_t N>
shared_ptr<Buffer>createSidesBuffer(shared_ptr<Adjacency const>const&adj){
  vector<GPUEdgeData<N>>dst(adj->getNofEdges());
  auto const src = reinterpret_cast<Vertex3Df const*>(adj->getVertices().data());

  writeEdges<N>(dst,src,adj);
  return make_shared<ge::gl::Buffer>(dst);
}

template<size_t N>
shared_ptr<VertexArray>createVAO(shared_ptr<Buffer>const&sides){
  auto vao = make_shared<VertexArray>();
  GLenum const normalized = GL_FALSE;
  GLuint const divisor = N;
  GLsizei const stride = GLsizei(sizeof(GPUEdgeData<N>));
  vao->addAttrib(sides,0,sizeof(GPUEdgeData<N>::vertexA    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<N>,vertexA    ),normalized,divisor);
  vao->addAttrib(sides,1,sizeof(GPUEdgeData<N>::vertexB    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<N>,vertexB    ),normalized,divisor);
  vao->addAttrib(sides,2,sizeof(GPUEdgeData<N>::nofOpposite)/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<N>,nofOpposite),normalized,divisor);
  for(GLuint o=0;o<N;++o){
    vao->addAttrib(sides,3+o,3,GL_FLOAT,stride,offsetof(GPUEdgeData<N>,oppositeVertices)+o*sizeof(GPUEdgeData<N>::oppositeVertices[0]),normalized,divisor);
  }
  return vao;
}

shared_ptr<Program>createProgram(vars::Vars&vars){
#include<VSSV/DrawSidesUsingPointsShader.h>
#include<SilhouetteShaders.h>

  auto program = make_shared<ge::gl::Program>(
      make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        vars.getBool("vssv.usePlanes"             )?Shader::define("USE_PLANES"               ):"",
        vars.getBool("vssv.useStrips"             )?Shader::define("USE_TRIANGLE_STRIPS"      ):"",
        vars.getBool("vssv.useAllOppositeVertices")?Shader::define("USE_ALL_OPPOSITE_VERTICES"):"",
        silhouetteFunctions,
        vertexShaderSrc));

  return program;
}

DrawSidesUsingPoints::DrawSidesUsingPoints(vars::Vars&vars,shared_ptr<Adjacency const>const&adj):vars(vars){
  maxMultiplicity = adj->getMaxMultiplicity();
  nofEdges        = adj->getNofEdges();
  if(maxMultiplicity == 2){
    sides    = createSidesBuffer<2>(adj);
    vao      = createVAO<2>(sides);
  }else{
    throw runtime_error("VSSV - unsupported max multiplicity");
  }
  program  = createProgram(vars);
}

void DrawSidesUsingPoints::draw(
    vec4 const&light     ,
    mat4 const&view      ,
    mat4 const&projection){
  program->use();
  program->setMatrix4fv("viewMatrix"      ,value_ptr(view      ));
  program->setMatrix4fv("projectionMatrix",value_ptr(projection));
  program->set4fv      ("lightPosition"   ,value_ptr(light     ));
  vao->bind();
  if(vars.getBool("vssv.useStrips"))
    glDrawArraysInstanced(GL_TRIANGLE_STRIP,0,4,GLsizei(nofEdges*maxMultiplicity));
  else
    glDrawArraysInstanced(GL_TRIANGLES     ,0,6,GLsizei(nofEdges*maxMultiplicity));
  vao->unbind();
}