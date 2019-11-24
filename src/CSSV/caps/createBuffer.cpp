#include<Vars/Vars.h>
#include<geGL/geGL.h>

#include<FunctionPrologue.h>
#include<FastAdjacency.h>
#include<Simplex.h>

#include<CSSV/caps/createBuffer.h>


using namespace std;
using namespace ge::gl;

void cssv::createCapsBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","adjacency");

  auto adj = vars.get<Adjacency>("adjacency");

  auto copyTriangles = [](Triangle4Df *const dst,Triangle3Df const*const src,size_t nofTriangles){
    for(size_t t=0;t<nofTriangles;++t)
      createHomogenous(dst[t],src[t]);
  };

  auto const nofTriangles = adj->getNofTriangles();
  vector<Triangle4Df>dst(nofTriangles);
  auto const dstPtr = dst.data();
  auto const srcPtr = reinterpret_cast<Triangle3Df const*>(adj->getVertices().data());
  copyTriangles(dstPtr,srcPtr,nofTriangles);

  vars.reCreate<Buffer>("cssv.method.caps.buffer",dst);
}
