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

#include <FastAdjacency.h>

namespace cssv::sides{


void createExtractSilhouetteMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.param.dontUsePlanes"
      ,"cssv.param.dontUseInterleaving"
      ,"cssv.param.alignment"
      ,"cssv.param.dontExtractMultiplicity"
      ,"cssv.param.dontPackMult"
      ,"adjacency"
      );
  if(!vars.getBool("cssv.param.dontUsePlanes")){
    if(!vars.getBool("cssv.param.dontUseInterleaving"))
      createInterleavedPlanesEdges(vars);
    else
      createPlanesEdges(vars);
  }else{
    createBasicEdges(vars);
  }
  createSilhouetteBuffer(vars);

  if(!vars.getBool("cssv.param.dontExtractMultiplicity")){
    auto const adj = vars.get<Adjacency>("adjacency");
    if(vars.getBool("cssv.param.dontPackMult"))
      vars.reCreate<Buffer>("cssv.method.multBuffer",sizeof(uint32_t)*2*adj->getNofEdges());
    else
      vars.reCreate<Buffer>("cssv.method.multBuffer",sizeof(uint32_t)*adj->getNofEdges());
  }
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

  bool const dontExtMult = vars.getBool("cssv.param.dontExtractMultiplicity");
  if(!dontExtMult){
    auto multBuffer = vars.get<Buffer>("cssv.method.multBuffer");
    auto edgeBuffer = vars.get<Buffer>("cssv.method.edgeBuffer");

    //std::vector<int>mult;
    //multBuffer->getData(mult);
    //std::cerr << "########" << std::endl;
    //for(auto const&x:mult)
    //  std::cerr << x << std::endl;
    //std::cerr << "########" << std::endl;

    //std::vector<float>ed;
    //edgeBuffer->getData(ed);
    //std::cerr << "((((((((" << std::endl;
    //for(size_t e=0;e<ed.size();e+=6){
    //  for(int i=0;i<2;++i){
    //    for(int j=0;j<3;++j)
    //      std::cerr << ed[e+i*3+j] << " ";
    //    std::cerr << " - ";
    //  }
    //  std::cerr << std::endl;
    //}
    //std::cerr << "))))))))" << std::endl;
    

    multBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,3);
    edgeBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  }

  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp          ))
    ->set4fv      ("lightPosition",value_ptr(lightPosition));

  if(dontExtMult){
    glPatchParameteri(GL_PATCH_VERTICES,2);
    glDrawArraysIndirect(GL_PATCHES,NULL);
  }else{
    glDrawArraysIndirect(GL_POINTS,NULL);
  }
  vao->unbind();

  // */
}
