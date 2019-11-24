#include <geGL/geGL.h>
#include <Vars/Vars.h>
#include <CSSV/drawSides.h>
#include <CSSV/createDrawSidesProgram.h>
#include <CSSV/createSidesVAO.h>
#include<CSSV/extractSilhouettes.h>
#include<CSSV/createDIBO.h>
#include<CSSV/createBasicEdges.h>
#include<CSSV/createSilhouetteBuffer.h>
#include<CSSV/createPlanesEdges.h>
#include<CSSV/createInterleavedPlanesEdges.h>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/StaticCalls.h>
#include<FunctionPrologue.h>
#include<createAdjacency.h>
#include<ifExistStamp.h>

using namespace glm;
using namespace ge::gl;
using namespace std;

namespace cssv{
void createExtractSilhouetteMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.param.usePlanes"
      ,"cssv.param.useInterleaving"
      ,"cssv.param.alignment"
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
}


void cssv::drawSides(
    vars::Vars&vars,
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){

  createDIBO(vars);
  createAdjacency(vars);
  createExtractSilhouetteMethod(vars);

  extractSilhouettes(vars,lightPosition);
  ifExistStamp(vars,"compute");

  cssv::createSidesVAO(vars);
  cssv::createDrawSidesProgram(vars);

  auto dibo    = vars.get<Buffer     >("cssv.method.dibo"             );
  auto vao     = vars.get<VertexArray>("cssv.method.sides.vao"        );
  auto program = vars.get<Program    >("cssv.method.sides.drawProgram");

  auto mvp = projectionMatrix * viewMatrix;
  dibo->bind(GL_DRAW_INDIRECT_BUFFER);
  vao->bind();
  program->use();
  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp          ))
    ->set4fv      ("lightPosition",value_ptr(lightPosition));
  glPatchParameteri(GL_PATCH_VERTICES,2);
  glDrawArraysIndirect(GL_PATCHES,NULL);
  vao->unbind();
}
