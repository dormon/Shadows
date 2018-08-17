#include <VSSV/DrawSidesUsingAllPlanes.h>

#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <FastAdjacency.h>
#include <Simplex.h>
#include <ShadowMethod.h>
#include <array>


using namespace std;
using namespace ge::gl;
using namespace glm;

namespace vssvUsingAllPlanes{

#define USE_TUPLE

#ifdef USE_TUPLE
template<size_t N>
using GPUEdgeData = tuple<Vertex3Df,Vertex3Df,array<Vertex4Df,N>>;

enum GPUEdgeDataParts{
  VERTEX_A = 0,
  VERTEX_B = 1,
  PLANES   = 2,
};
#else
template<size_t N>
struct GPUEdgeData{
  Vertex3Df vertexA;
  Vertex3Df vertexB;
  array<Vertex4Df,N>planes;
};
#endif

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
    writePlane(planes[o],src,e,o,adj);
}

template<size_t N>
void writeEdge(GPUEdgeData<N>&edge,Vertex3Df const*const src,size_t e,shared_ptr<Adjacency const>const&adj){
#ifdef USE_TUPLE
  get<VERTEX_A>(edge) = src[adj->getEdge(e,0)/3];
  get<VERTEX_B>(edge) = src[adj->getEdge(e,1)/3];
  writePlanes(get<PLANES>(edge),src,e,adj);
#else
  edge.vertexA = src[adj->getEdge(e,0)/3];
  edge.vertexB = src[adj->getEdge(e,1)/3];
  writePlanes(edge.planes,src,e,adj);
#endif
}

template<size_t N>
void writeEdges(vector<GPUEdgeData<N>>&dst,Vertex3Df const*const src,shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst[e],src,e,adj);
}

template<size_t N>
shared_ptr<Buffer>createSidesBuffer(shared_ptr<Adjacency const>const&adj){
  std::cout << sizeof(GPUEdgeData<N>) << std::endl;
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
  vao->addAttrib(sides,0,/*sizeof(GPUEdgeData<N>::vertexA    )/sizeof(float)*/3,GL_FLOAT,stride,/*offsetof(GPUEdgeData<N>,vertexA    )*/0,normalized,N);
  vao->addAttrib(sides,1,/*sizeof(GPUEdgeData<N>::vertexB    )/sizeof(float)*/3,GL_FLOAT,stride,/*offsetof(GPUEdgeData<N>,vertexB    )*/sizeof(float)*3,normalized,N);
  for(GLuint o=0;o<N;++o)
    vao->addAttrib(sides,2+o,4,GL_FLOAT,stride,/*offsetof(GPUEdgeData<N>,planes)+sizeof(Vertex4Df)*o*/sizeof(float)*6+o*sizeof(float)*4,normalized,N);
  return vao;
}

shared_ptr<Program>createProgram(vars::Vars const&vars,shared_ptr<Adjacency const>const&adj){
#include<VSSV/DrawSidesUsingAllPlanesShader.h>
#include<SilhouetteShaders.h>

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

