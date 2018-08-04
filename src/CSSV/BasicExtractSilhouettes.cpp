#include <CSSV/BasicExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>

using namespace cssv;

size_t getFloatsPerEdge(std::shared_ptr<Adjacency const>const&adj){
  size_t const verticesPerEdgeData = verticesPerEdge + adj->getMaxMultiplicity();
  return verticesPerEdgeData * componentsPerVertex4D;
}

size_t getDstVerticesOffset(size_t e,std::shared_ptr<Adjacency const>const&adj){
  return e * getFloatsPerEdge(adj);
}

void copyVertex3D(float*const dst,float const*const src){
  size_t const sizeofVertex3DInBytes = componentsPerVertex3D * sizeof(float);
  std::memcpy(dst,src,sizeofVertex3DInBytes);
}

void writeVertexA(float*dstPtr,float const*srcPtr,size_t e,std::shared_ptr<Adjacency const>const&adj){
  auto const vertexDstPtr = dstPtr + getDstVerticesOffset(e,adj);
  auto const vertexSrcPtr = srcPtr + adj->getEdgeVertexA(e);
  copyVertex3D(vertexDstPtr,vertexSrcPtr);
  vertexDstPtr[3] = static_cast<float>(adj->getNofOpposite(e));
}

void writeVertexB(float*dstPtr,float const*srcPtr,size_t e,std::shared_ptr<Adjacency const>const&adj){
  auto const vertexDstPtr = dstPtr + getDstVerticesOffset(e,adj) + componentsPerVertex4D;
  auto const vertexSrcPtr = srcPtr + adj->getEdgeVertexB(e);
  copyVertex3D(vertexDstPtr,vertexSrcPtr);
  vertexDstPtr[3] = 1.f;
}

void writeEmptyOppositeVertices(float*dstPtr,size_t e,std::shared_ptr<Adjacency const>const&adj){
  size_t  const nofEmptyOppositeVertices = adj->getMaxMultiplicity() - adj->getNofOpposite(e);
  auto    const emptyOppositeVerticesDstPtr = dstPtr + componentsPerVertex4D*adj->getNofOpposite(e);
  uint8_t const value = 0;
  std::memset(emptyOppositeVerticesDstPtr,value,nofEmptyOppositeVertices*componentsPerVertex4D*sizeof(float));
}

void writeOppositeVertex(float*dstPtr,float const*srcPtr,size_t o,size_t e,std::shared_ptr<Adjacency const>const&adj){
  auto const oppositeVertexDstPtr = dstPtr + o*componentsPerVertex4D;
  auto const oppositeVertexSrcPtr = srcPtr + adj->getOpposite(e,o);
  copyVertex3D(oppositeVertexDstPtr,oppositeVertexSrcPtr);
  oppositeVertexDstPtr[3]=1.f;
}

void writeUsedOppositeVertices(float*dstPtr,float const*srcPtr,size_t e,std::shared_ptr<Adjacency const>const&adj){
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    writeOppositeVertex(dstPtr,srcPtr,o,e,adj);
}

void writeOppositeVertices(float*dstPtr,float const*srcPtr,size_t e,std::shared_ptr<Adjacency const>const&adj){
  auto const oppositeVerticesDstPtr = dstPtr + e*getFloatsPerEdge(adj) + componentsPerVertex4D * 2;
  writeUsedOppositeVertices(oppositeVerticesDstPtr,srcPtr,e,adj);
  writeEmptyOppositeVertices(oppositeVerticesDstPtr,e,adj);
}

void writeEdge(float*dst,float const*src,size_t e,std::shared_ptr<Adjacency const>const&adj){
  writeVertexA(dst,src,e,adj);
  writeVertexB(dst,src,e,adj);
  writeOppositeVertices(dst,src,e,adj);
}

void writeEdges(float*dst,float const*src,std::shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst,src,e,adj);
}

std::shared_ptr<ge::gl::Buffer>createEdgeBuffer(std::shared_ptr<Adjacency const>const&adj){
  size_t const verticesPerEdgeData = verticesPerEdge+adj->getMaxMultiplicity();
  std::vector<float>dst(componentsPerVertex4D*verticesPerEdgeData*adj->getNofEdges());
  auto const srcData = adj->getVertices();
  auto const src = srcData.data();
  writeEdges(dst.data(),src,adj);
  return std::make_shared<ge::gl::Buffer>(dst);
}

std::shared_ptr<ge::gl::Buffer>createSillouetteBuffer(std::shared_ptr<Adjacency const>const&adj){
  size_t const bufferSize = sizeof(float)*componentsPerVertex4D*verticesPerQuad*adj->getNofEdges()*adj->getMaxMultiplicity();
  auto sillhouettes=std::make_shared<ge::gl::Buffer>(bufferSize,nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
  return sillhouettes;
}

BasicExtractSilhouettes::BasicExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj):ExtractSilhouettes(vars,adj){
  nofEdges     = adj->getNofEdges();
  edges        = createEdgeBuffer(adj);
  sillhouettes = createSillouetteBuffer(adj);
}

