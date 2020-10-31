#include <VSSV/DrawSidesUsingAllPlanes.h>
#include <Vars/Vars.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>
#include <ShadowMethod.h>
#include <array>
#include <VSSV/DrawSidesUsingAllPlanesShader.h>
#include <SilhouetteShaders.h>


using namespace std;
using namespace ge::gl;
using namespace glm;

namespace vssvUsingAllPlanes{

template<size_t N>
struct GPUEdgeData{
  Vertex3Df         vertexA;
  Vertex3Df         vertexB;
  array<Vertex4Df,N>planes ;
};

void writePlane(Vertex4Df&plane,Vertex3Df const*const src,size_t e,size_t o,shared_ptr<Adjacency const>const&adj){
  auto const glmPlane = computePlane(
          toVec3(src[adj->getEdge    (e,0)/3].elements),
          toVec3(src[adj->getEdge    (e,1)/3].elements),
          toVec3(src[adj->getOpposite(e,o)/3].elements));
  for(int i=0;i<4;++i)
    plane.elements[i] = glmPlane[i];
}

template<size_t N>
void writePlanes(array<Vertex4Df,N>&planes,Vertex3Df const*const src,size_t e,shared_ptr<Adjacency const>const&adj){
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    writePlane(planes.at(o),src,e,o,adj);
}

template<size_t N>
void writeEdge(GPUEdgeData<N>&edge,Vertex3Df const*const src,size_t e,shared_ptr<Adjacency const>const&adj){
  edge.vertexA = src[adj->getEdge(e,0)/3];
  edge.vertexB = src[adj->getEdge(e,1)/3];
  writePlanes(edge.planes,src,e,adj);
}

template<size_t N>
void writeEdges(vector<GPUEdgeData<N>>&dst,Vertex3Df const*const src,shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst.at(e),src,e,adj);
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
  auto   const stride = static_cast<GLsizei>(sizeof(GPUEdgeData<N>));
  vao->addAttrib(sides,0,sizeof(GPUEdgeData<N>::vertexA    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<N>,vertexA    ),normalized,N);
  vao->addAttrib(sides,1,sizeof(GPUEdgeData<N>::vertexB    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<N>,vertexB    ),normalized,N);
  for(GLuint o=0;o<N;++o)
    vao->addAttrib(sides,2+o,4,GL_FLOAT,stride,offsetof(GPUEdgeData<N>,planes)+sizeof(Vertex4Df)*o,normalized,N);
  return vao;
}

shared_ptr<Program>createProgram(vars::Vars const&vars,shared_ptr<Adjacency const>const&adj){
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        vars.getBool("vssv.useStrips")?Shader::define("USE_TRIANGLE_STRIPS"):"",
        Shader::define("MAX_MULTIPLICITY",static_cast<uint32_t>(adj->getMaxMultiplicity())),
        silhouetteFunctions,
        vertexShaderSrc));
  return program;
}
}

DrawSidesUsingAllPlanes::DrawSidesUsingAllPlanes(vars::Vars&vars,shared_ptr<Adjacency const>const&adj):DrawSides(vars){
  maxMultiplicity = adj->getMaxMultiplicity();
  if(maxMultiplicity == 2){
    sides = vssvUsingAllPlanes::createSidesBuffer<2>(adj  );
    vao   = vssvUsingAllPlanes::createVAO        <2>(sides);
  }else{
    throw runtime_error("VSSV - DrawSidesUsingAllPlanes unsupported max multiplicity");
  }
  program = vssvUsingAllPlanes::createProgram(vars,adj);
  nofEdges = adj->getNofEdges();
}

