#include<CSSV/CSSV.h>
#include<FastAdjacency.h>
#include<util.h>
#include<geGL/StaticCalls.h>

#include<CSSV/DrawCapsProgram.h>
#include<CSSV/DrawSidesProgram.h>
#include<CSSV/BasicExtractSilhouettes.h>
#include<CSSV/PlanesExtractSilhouettes.h>
#include<CSSV/InterleavedPlanesExtractSilhouettes.h>
#include<CSSV/DrawCaps.h>
#include<FunctionPrologue.h>

using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;

void createExtractSilhouetteMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.param.usePlanes","cssv.param.useInterleaving");
  auto const adj = createAdjacencyBase(vars);
  if(vars.getBool("cssv.param.usePlanes")){
    if(vars.getBool("cssv.param.useInterleaving"))
      vars.reCreate<InterleavedPlanesExtractSilhouettes>("cssv.method.extractSilhouettes",vars,adj);
    else
      vars.reCreate<PlanesExtractSilhouettes>("cssv.method.extractSilhouettes",vars,adj);
  }else
    vars.reCreate<BasicExtractSilhouettes>("cssv.method.extractSilhouettes",vars,adj);
}

void createDrawSides(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.method.extractSilhouettes");
  auto ex = vars.getReinterpret<ExtractSilhouettes>("cssv.method.extractSilhouettes");
  vars.reCreate<DrawSides>("cssv.method.drawSides",ex->sillhouettes,ex->dibo);
}

void createDrawCaps(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method");
  auto const adj = createAdjacencyBase(vars);
  vars.reCreate<DrawCaps>("cssv.method.drawCaps",adj);
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
  createExtractSilhouetteMethod(vars);
  createDrawSides(vars);
  auto ex = vars.getReinterpret<ExtractSilhouettes>("cssv.method.extractSilhouettes");
  ex->compute(lightPosition);
  ifExistStamp("compute");
  auto sides = vars.get<DrawSides>("cssv.method.drawSides");
  sides->draw(lightPosition,viewMatrix,projectionMatrix);
}

void CSSV::drawCaps(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  createDrawCaps(vars);
  auto caps = vars.get<DrawCaps>("cssv.method.drawCaps");
  caps->draw(lightPosition,viewMatrix,projectionMatrix);
}

