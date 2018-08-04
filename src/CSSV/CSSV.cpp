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

using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;

unique_ptr<ExtractSilhouettes>createExtractSilhouetteMethod(vars::Vars&vars,shared_ptr<Adjacency const>const&adj){
  if(vars.getBool("cssv.usePlanes")){
    if(vars.getBool("cssv.useInterleaving"))
      return make_unique<InterleavedPlanesExtractSilhouettes>(vars,adj);
    else
      return make_unique<PlanesExtractSilhouettes>(vars,adj);
  }else
    return make_unique<BasicExtractSilhouettes>(vars,adj);
}

CSSV::CSSV(vars::Vars&vars):
  ShadowVolumes(vars  )
{
  auto const adj = createAdjacency(vars);
  extractSilhouettes = createExtractSilhouetteMethod(vars,adj);
  caps = make_unique<DrawCaps>(adj);
  sides = make_unique<DrawSides>(extractSilhouettes->sillhouettes,extractSilhouettes->dibo);
}

CSSV::~CSSV(){}

void CSSV::drawSides(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  extractSilhouettes->compute(lightPosition);
  ifExistStamp("compute");
  sides->draw(lightPosition,viewMatrix,projectionMatrix);
}

void CSSV::drawCaps(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  caps->draw(lightPosition,viewMatrix,projectionMatrix);
}

