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
#include<createAdjacency.h>

using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;

void createExtractSilhouetteMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.param.usePlanes","cssv.param.useInterleaving","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");
  if(vars.getBool("cssv.param.usePlanes")){
    if(vars.getBool("cssv.param.useInterleaving"))
      vars.reCreate<InterleavedPlanesExtractSilhouettes>("cssv.method.extractSilhouettes",vars);
    else
      vars.reCreate<PlanesExtractSilhouettes>("cssv.method.extractSilhouettes",vars);
  }else
    vars.reCreate<BasicExtractSilhouettes>("cssv.method.extractSilhouettes",vars);
}

void createDrawSides(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.method.extractSilhouettes");
  auto ex = vars.getReinterpret<ExtractSilhouettes>("cssv.method.extractSilhouettes");
  vars.reCreate<DrawSides>("cssv.method.drawSides",ex->sillhouettes,ex->dibo);
}

void createDrawCaps(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");
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
  createAdjacency(vars);
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

