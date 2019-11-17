#include <CSSV/PlanesExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>
#include <FunctionPrologue.h>
#include <CSSV/createPlanesEdges.h>
#include <CSSV/createSilhouetteBuffer.h>

using namespace cssv;


PlanesExtractSilhouettes::PlanesExtractSilhouettes(vars::Vars&vars):ExtractSilhouettes(vars){
  assert(this!=nullptr);

  createPlanesEdges(vars);
  createSilhouetteBuffer(vars);
}

