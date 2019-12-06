#include<CSSV/sides/createBasicEdges.h>
#include<FunctionPrologue.h>
#include<FastAdjacency.h>
#include<Simplex.h>
#include<geGL/geGL.h>

void cssv::sides::createBasicEdges(vars::Vars&vars){
  //FUNCTION_PROLOGUE("cssv.method","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");

  auto verts = adj->getVertices().data();
  auto const nV = 2+adj->getMaxMultiplicity();
  std::vector<float>dst(adj->getNofEdges() * nV *4);

  for(size_t e=0;e<adj->getNofEdges();++e){
    for(int i=0;i<3;++i)
      dst[e*(nV*4)+0*4+i] = verts[adj->getEdgeVertexA(e)+i];
    dst[e*(nV*4)+0*4+3] = adj->getNofOpposite(e);

    for(int i=0;i<3;++i)
      dst[e*(nV*4)+1*4+i] = verts[adj->getEdgeVertexB(e)+i];
    dst[e*(nV*4)+1*4+3] = 1;

    for(uint32_t o=0;o<adj->getNofOpposite(e);++o){
      for(int i=0;i<3;++i)
        dst[e*(nV*4)+(2+o)*4+i] = verts[adj->getOpposite(e,o)+i];
      dst[e*(nV*4)+(2+o)*4+3] = 1;
    }

    for(uint32_t o=adj->getNofOpposite(e);o<adj->getMaxMultiplicity();++o){
      for(int i=0;i<4;++i)
        dst[e*(nV*4)+(2+o)*4+i] = 0;
    }
  }
  vars.reCreate<ge::gl::Buffer>("cssv.method.edges",dst);
}
