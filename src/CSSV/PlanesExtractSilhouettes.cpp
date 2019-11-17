#include <CSSV/PlanesExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>

using namespace cssv;

PlanesExtractSilhouettes::PlanesExtractSilhouettes(vars::Vars&vars):ExtractSilhouettes(vars){
  assert(this!=nullptr);
  auto const adj = vars.get<Adjacency>("adjacency");
  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;
  edges = std::make_shared<ge::gl::Buffer>(adj->getNofEdges()*floatsPerEdge*sizeof(float));

  auto const dstPtr = static_cast<float      *>(edges->map());
  auto const srcPtr = adj->getVertices().data();

  size_t const sizeofVertex3DInBytes = componentsPerVertex3D * sizeof(float);
  for(size_t edgeIndex=0;edgeIndex<adj->getNofEdges();++edgeIndex){
    auto const edgeDstPtr    = dstPtr + edgeIndex*floatsPerEdge;
    auto const vertexADstPtr = edgeDstPtr;
    auto const vertexBDstPtr = vertexADstPtr + componentsPerVertex3D;
    auto const planesDstPtr  = vertexBDstPtr + componentsPerVertex3D;

    auto const vertexASrcPtr = srcPtr + adj->getEdgeVertexA(edgeIndex);
    auto const vertexBSrcPtr = srcPtr + adj->getEdgeVertexB(edgeIndex);

    std::memcpy(vertexADstPtr,vertexASrcPtr,sizeofVertex3DInBytes);
    std::memcpy(vertexBDstPtr,vertexBSrcPtr,sizeofVertex3DInBytes);
    size_t const sizeofPlane3DInBytes = componentsPerPlane3D*sizeof(float);
    for(size_t oppositeIndex=0;oppositeIndex<adj->getNofOpposite(edgeIndex);++oppositeIndex){
      auto const planeDstPtr          = planesDstPtr + oppositeIndex*componentsPerPlane3D;
      auto const oppositeVertexSrcPtr = srcPtr + adj->getOpposite(edgeIndex,oppositeIndex);
      auto const plane                = computePlane(toVec3(vertexASrcPtr),toVec3(vertexBSrcPtr),toVec3(oppositeVertexSrcPtr));
      std::memcpy(planeDstPtr,glm::value_ptr(plane),sizeofPlane3DInBytes);
    }
    size_t  const nofEmptyPlanes    = maxNofOppositeVertices - adj->getNofOpposite(edgeIndex);
    auto    const emptyPlanesDstPtr = planesDstPtr + componentsPerPlane3D*adj->getNofOpposite(edgeIndex);
    uint8_t const value             = 0;
    std::memset(emptyPlanesDstPtr,value,sizeofPlane3DInBytes*nofEmptyPlanes);
  }
  edges->unmap();
  nofEdges = adj->getNofEdges();

  sillhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(float)*componentsPerVertex4D*verticesPerQuad*nofEdges*adj->getMaxMultiplicity(),
      nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}

