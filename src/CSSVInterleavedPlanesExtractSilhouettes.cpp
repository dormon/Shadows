#include <CSSVInterleavedPlanesExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>

CSSVInterleavedPlanesExtractSilhouettes::CSSVInterleavedPlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj):CSSVExtractSilhouettes(vars,adj){
  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;
  edges = std::make_shared<ge::gl::Buffer>(adj->getNofEdges()*floatsPerEdge*sizeof(float));

  auto const dstPtr = static_cast<float      *>(edges->map());
  auto const srcPtr = static_cast<float const*>(adj->getVertices() );

  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*0] = srcPtr[adj->getEdgeVertexA(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*1] = srcPtr[adj->getEdgeVertexA(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*2] = srcPtr[adj->getEdgeVertexA(e)+2];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*3] = srcPtr[adj->getEdgeVertexB(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*4] = srcPtr[adj->getEdgeVertexB(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+adj->getNofEdges()*5] = srcPtr[adj->getEdgeVertexB(e)+2];
  for(size_t o=0;o<maxNofOppositeVertices;++o)
    for(size_t e=0;e<adj->getNofEdges();++e){
      glm::vec4 plane = glm::vec4(0.f);
      if(o<adj->getNofOpposite(e))
        plane = computePlane(toVec3(srcPtr+adj->getEdgeVertexA(e)),toVec3(srcPtr+adj->getEdgeVertexB(e)),toVec3(srcPtr+adj->getOpposite(e,o)));
      for(size_t k=0;k<componentsPerPlane3D;++k)
        dstPtr[e+adj->getNofEdges()*(6+o*componentsPerPlane3D+k)] = plane[(uint32_t)k];
    }

  edges->unmap();
  nofEdges = adj->getNofEdges();

  sillhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(float)*componentsPerVertex4D*verticesPerQuad*nofEdges*adj->getMaxMultiplicity(),
      nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}

