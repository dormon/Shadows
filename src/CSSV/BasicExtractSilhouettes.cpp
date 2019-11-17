#include <CSSV/BasicExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>
#include <Simplex.h>
#include <FunctionPrologue.h>
#include <CSSV/createSilhouetteBuffer.h>
#include <CSSV/createBasicEdges.h>

using namespace cssv;

BasicExtractSilhouettes::BasicExtractSilhouettes(vars::Vars&vars):ExtractSilhouettes(vars){
  createBasicEdges(vars);

  createSilhouetteBuffer(vars);
}

