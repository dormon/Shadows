#include<CSSV/createBasicEdges.h>
#include<FunctionPrologue.h>
#include<FastAdjacency.h>
#include<Simplex.h>
#include<geGL/geGL.h>

template<size_t N>
struct GPUEdgeData{
  Line4Df   edge;
  Vertex4Df oppositeVertices[N];
};

struct Quad4Df{
  Vertex4Df data[4];
};

template<size_t N>
struct GPUSilhouetteData{
  Quad4Df quads[N];
};

void writeVertexA(Vertex4Df&dst,Vertex3Df const&src,size_t nofOpposite){
  dst = src;
  dst.elements[3] = static_cast<float>(nofOpposite);
}

void writeVertexB(Vertex4Df&dst,Vertex3Df const&src){
  createHomogenous(dst,src);
}

void writeEdge(Line4Df&dst,Vertex3Df const*const vertices,size_t e,Adjacency const*adj){
  writeVertexA(dst.vertices[0],vertices[adj->getEdgeVertexA(e)/3],adj->getNofOpposite(e));
  writeVertexB(dst.vertices[1],vertices[adj->getEdgeVertexB(e)/3]);
}

void writeOppositeVertex(Vertex4Df&dstPtr,Vertex3Df const*srcPtr,size_t o,size_t e,Adjacency const*adj){
  createHomogenous(dstPtr,srcPtr[adj->getOpposite(e,o)/3]);
}

void writeUsedOppositeVertices(Vertex4Df*dstPtr,Vertex3Df const*const srcPtr,size_t e,Adjacency const*adj){
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    writeOppositeVertex(dstPtr[o],srcPtr,o,e,adj);
}

template<size_t N=2>
void writeEmptyOppositeVertices(Vertex4Df*dstPtr,size_t e,Adjacency const*adj){
  for(size_t o=adj->getNofOpposite(e);o<N;++o)
    dstPtr[o].clear();
}

void writeOppositeVertices(Vertex4Df*const dstPtr,Vertex3Df const* const srcPtr,size_t e,Adjacency const*adj){
  writeUsedOppositeVertices (dstPtr,srcPtr,e,adj);
  writeEmptyOppositeVertices(dstPtr,e,adj);
}

template<size_t N=2>
void writeEdgeData(GPUEdgeData<N>&dst,Vertex3Df const*const src,size_t e,Adjacency const*adj){
  writeEdge(dst.edge,src,e,adj);
  writeOppositeVertices(dst.oppositeVertices,src,e,adj);
}

template<size_t N=2>
void writeEdges(GPUEdgeData<N>*const dst,Vertex3Df const*const src,Adjacency const*adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdgeData<N>(dst[e],src,e,adj);
}

template<size_t N=2>
std::shared_ptr<ge::gl::Buffer>createEdgeBuffer(Adjacency const*adj){
  auto const src = reinterpret_cast<Vertex3Df const*>(adj->getVertices().data());
  std::vector<GPUEdgeData<N>>dst(adj->getNofEdges());
  writeEdges<N>(dst.data(),src,adj);
  return std::make_shared<ge::gl::Buffer>(dst);
}

template<size_t N=2>
std::shared_ptr<ge::gl::Buffer>createSillouetteBuffer(Adjacency const*adj){
  size_t const bufferSize = sizeof(GPUSilhouetteData<N>)*adj->getNofEdges();
  auto sillhouettes=std::make_shared<ge::gl::Buffer>(bufferSize,nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
  return sillhouettes;
}


void cssv::createBasicEdges(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");
  if(adj->getMaxMultiplicity() == 2){
    auto verts = adj->getVertices().data();
    auto const src = reinterpret_cast<Vertex3Df const*>(verts);
    std::vector<GPUEdgeData<2>>dst(adj->getNofEdges());
    writeEdges<2>(dst.data(),src,adj);
    vars.reCreate<ge::gl::Buffer>("cssv.method.edges",dst);
  }else{
    throw std::runtime_error("unsupported max multiplicity");
  }


}
