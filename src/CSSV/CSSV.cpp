#include<CSSV/CSSV.h>
#include<FastAdjacency.h>
#include<util.h>
#include<geGL/StaticCalls.h>
#include <geGL/geGL.h>
#include <glm/glm.hpp>
#include <Model.h>
#include <TimeStamp.h>

#include<CSSV/DrawSides.h>
#include<CSSV/DrawCaps.h>
#include<CSSV/DrawCapsProgram.h>
#include<CSSV/DrawSidesProgram.h>
#include<CSSV/DrawCaps.h>
#include<CSSV/extractSilhouettes.h>
#include<CSSV/createDIBO.h>
#include<CSSV/createBasicEdges.h>
#include<CSSV/createSilhouetteBuffer.h>
#include<CSSV/createPlanesEdges.h>
#include<CSSV/createInterleavedPlanesEdges.h>
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

void createDrawSides(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.method.silhouettes"
      ,"cssv.method.dibo"
      );
  vars.reCreate<DrawSides>("cssv.method.drawSides",
      vars.get<Buffer>("cssv.method.silhouettes" ),
      vars.get<Buffer>("cssv.method.dibo"        )
      );
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
  createDIBO(vars);
  createAdjacency(vars);
  createExtractSilhouetteMethod(vars);
  createDrawSides(vars);

  extractSilhouettes(vars,lightPosition);
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

