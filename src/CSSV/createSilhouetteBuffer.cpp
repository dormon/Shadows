#include <CSSV/createSilhouetteBuffer.h>
#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <FastAdjacency.h>
#include <geGL/geGL.h>
#include <ShadowMethod.h>

void cssv::createSilhouetteBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");
  auto nofEdges = adj->getNofEdges();
  auto silhouettes = vars.reCreate<ge::gl::Buffer>("cssv.method.silhouettes",
      sizeof(float)*componentsPerVertex4D*verticesPerQuad*nofEdges*adj->getMaxMultiplicity(),
      nullptr,GL_DYNAMIC_COPY);
  silhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}
