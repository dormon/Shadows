#include <CSSV/InterleavedPlanesExtractSilhouettes.h>
#include <geGL/StaticCalls.h>
#include <util.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>
#include<CSSV/InterleavedPlanesShader.h>
#include<CSSV/createExtractProgram.h>
#include<SilhouetteShaders.h>
#include <FunctionPrologue.h>
#include <CSSV/createInterleavedPlanesEdges.h>
#include <CSSV/createSilhouetteBuffer.h>

using namespace ge::gl;
using namespace std;
using namespace cssv;

InterleavedPlanesExtractSilhouettes::InterleavedPlanesExtractSilhouettes(vars::Vars&vars):ExtractSilhouettes(vars){

  createInterleavedPlanesEdges(vars);
  createSilhouetteBuffer(vars);
}

