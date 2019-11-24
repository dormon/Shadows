#include<CSSV/CSSV.h>
#include<FastAdjacency.h>
#include<util.h>
#include<geGL/StaticCalls.h>
#include <geGL/geGL.h>
#include <glm/glm.hpp>
#include <Model.h>
#include <TimeStamp.h>

#include<CSSV/extractSilhouettes.h>
#include<CSSV/createDIBO.h>
#include<CSSV/createBasicEdges.h>
#include<CSSV/createSilhouetteBuffer.h>
#include<CSSV/createPlanesEdges.h>
#include<CSSV/createInterleavedPlanesEdges.h>
#include<CSSV/drawSides.h>
#include<CSSV/caps/drawCaps.h>
#include<FunctionPrologue.h>
#include<createAdjacency.h>

using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;

void createExtractSilhouetteMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.param.usePlanes"
      ,"cssv.param.useInterleaving"
      ,"adjacency"
      ,"cssv.param.alignment"
      );
  if(vars.getBool("cssv.param.usePlanes")){
    if(vars.getBool("cssv.param.useInterleaving"))
      createInterleavedPlanesEdges(vars);
    else
      createPlanesEdges(vars);
  }else
    createBasicEdges(vars);
  createSilhouetteBuffer(vars);
}


CSSV::CSSV(vars::Vars&vars):
  ShadowVolumes(vars  )
{
}

CSSV::~CSSV(){
  vars.erase("cssv.method");
}


void CSSV::drawSides(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  createDIBO(vars);
  createAdjacency(vars);
  createExtractSilhouetteMethod(vars);

  extractSilhouettes(vars,lightPosition);
  ifExistStamp("compute");
  cssv::drawSides(vars,lightPosition,viewMatrix,projectionMatrix);
}

void CSSV::drawCaps(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  cssv::drawCaps(vars,lightPosition,viewMatrix,projectionMatrix);
}

