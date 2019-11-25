#include<glm/gtc/type_ptr.hpp>

#include<geGL/geGL.h>
#include<geGL/StaticCalls.h>
#include<Vars/Vars.h>

#include<FunctionPrologue.h>
#include<createAdjacency.h>
#include<ifExistStamp.h>

#include<CSSV/sides/draw.h>
#include<CSSV/sides/createDrawProgram.h>
#include<CSSV/sides/createVAO.h>
#include<CSSV/sides/extractSilhouettes.h>
#include<CSSV/sides/createDIBO.h>
#include<CSSV/sides/createBasicEdges.h>
#include<CSSV/sides/createSilhouetteBuffer.h>
#include<CSSV/sides/createPlanesEdges.h>
#include<CSSV/sides/createInterleavedPlanesEdges.h>

using namespace glm;
using namespace ge::gl;
using namespace std;

namespace cssv::sides{
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
  }else{
    createBasicEdges(vars);
  }
  createSilhouetteBuffer(vars);
}
}


void cssv::sides::draw(
    vars::Vars&vars,
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  FUNCTION_CALLER();

  createAdjacency(vars);
  createDIBO(vars);
  createExtractSilhouetteMethod(vars);

  extractSilhouettes(vars,lightPosition);
  ifExistStamp(vars,"compute");

  //*
  createVAO(vars);
  createDrawProgram(vars);

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

  // */
}
