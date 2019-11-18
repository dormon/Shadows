#include <CSSV/createInterleavedPlanesEdges.h>
#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <geGL/geGL.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>
#include <util.h>

using namespace ge::gl;

void cssv::createInterleavedPlanesEdges(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","adjacency","cssv.param.alignment");

  auto const adj = vars.get<Adjacency>("adjacency");
  size_t const alignSize = vars.getSizeT ("cssv.param.alignment");

  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;

  size_t floatAlign = align(alignSize , sizeof(float));
  size_t const bufferSize = align(adj->getNofEdges(),floatAlign)*floatsPerEdge;


  auto const src = adj->getVertices().data();

  std::vector<float>dst(bufferSize);
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*0] = src[adj->getEdgeVertexA(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*1] = src[adj->getEdgeVertexA(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*2] = src[adj->getEdgeVertexA(e)+2];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*3] = src[adj->getEdgeVertexB(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*4] = src[adj->getEdgeVertexB(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*5] = src[adj->getEdgeVertexB(e)+2];
  for(size_t o=0;o<maxNofOppositeVertices;++o)
    for(size_t e=0;e<adj->getNofEdges();++e){
      glm::vec4 plane = glm::vec4(0.f);
      if(o<adj->getNofOpposite(e))
        plane = computePlane(toVec3(src+adj->getEdgeVertexA(e)),toVec3(src+adj->getEdgeVertexB(e)),toVec3(src+adj->getOpposite(e,o)));
      for(size_t k=0;k<componentsPerPlane3D;++k)
        dst[e+align(adj->getNofEdges(),floatAlign)*(6+o*componentsPerPlane3D+k)] = plane[(uint32_t)k];
    }

  vars.reCreate<Buffer>("cssv.method.edges",dst);
}

