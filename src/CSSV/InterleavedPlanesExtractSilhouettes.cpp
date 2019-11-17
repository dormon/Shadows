#include <CSSV/InterleavedPlanesExtractSilhouettes.h>
#include <geGL/StaticCalls.h>
#include <util.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>
#include<CSSV/InterleavedPlanesShader.h>
#include<CSSV/createExtractProgram.h>
#include<SilhouetteShaders.h>

using namespace ge::gl;
using namespace std;
using namespace cssv;

InterleavedPlanesExtractSilhouettes::InterleavedPlanesExtractSilhouettes(vars::Vars&vars):ExtractSilhouettes(vars){

  auto const adj = vars.get<Adjacency>("adjacency");
  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;

  size_t const alignSize = vars.getSizeT ("cssv.param.alignment");
  size_t floatAlign = align(alignSize , sizeof(float));
  //size_t const floatAlign = align(vars.getSizeT("cssv.param.alignment") , sizeof(float)) / sizeof(float) ;
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

  edges = std::make_shared<Buffer>(dst);


  nofEdges = adj->getNofEdges();

  sillhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(float)*componentsPerVertex4D*verticesPerQuad*nofEdges*adj->getMaxMultiplicity(),
      nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}

/*
void InterleavedPlanesExtractSilhouettes::compute(glm::vec4 const&lightPosition){
  cssv::createExtractProgram(vars);
  auto const bufferSize = edges->getSize();
  auto const gigabyte = 1024*1024*1024;
  auto const bufferSizeInGigabytes = static_cast<double>(bufferSize) / static_cast<double>(gigabyte);
  dibo->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  auto program = vars.get<Program>("cssv.method.extractProgram");
  program
    ->set1ui    ("numEdge"           ,uint32_t(nofEdges)           )
    ->set4fv    ("lightPosition"     ,glm::value_ptr(lightPosition))
    ->bindBuffer("Edges"             ,edges                        )
    ->bindBuffer("Silhouettes"       ,sillhouettes                 )
    ->bindBuffer("DrawIndirectBuffer",dibo                         )
    ->dispatch((GLuint)getDispatchSize(nofEdges,vars.getUint32("cssv.param.computeSidesWGS")));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();

}

*/
