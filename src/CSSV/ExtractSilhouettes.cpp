#include <CSSV/ExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>
#include<CSSV/ExtractSilhouetteShader.h>
#include<SilhouetteShaders.h>
#include<FunctionPrologue.h>
#include<CSSV/createExtractProgram.h>
#include<CSSV/createDIBO.h>
#include<CSSV/extractSilhouettes.h>

using namespace std;
using namespace ge::gl;
using namespace cssv;

ExtractSilhouettes::ExtractSilhouettes(vars::Vars&vars):vars(vars){
  createDIBO(vars);
}


void ExtractSilhouettes::compute(glm::vec4 const&lightPosition){
  extractSilhouettes(vars,lightPosition);
}

