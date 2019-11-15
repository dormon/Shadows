#include<VSSV/VSSV.h>
#include<FastAdjacency.h>
#include<Simplex.h>
#include<VSSV/DrawSidesUsingPoints.h>
#include<VSSV/DrawSidesUsingPlanes.h>
#include<VSSV/DrawSidesUsingAllPlanes.h>
#include<Vars/Caller.h>

VSSV::VSSV(vars::Vars&vars):
  ShadowVolumes(vars       )
{
  vars::Caller caller(vars,__FUNCTION__);
  auto const adj = createAdjacencyBase(vars);

  if(vars.getBool("vssv.usePlanes")){
    if(vars.getBool("vssv.useAllOppositeVertices"))
      sides = std::make_unique<DrawSidesUsingAllPlanes>(vars,adj);
    else
      sides = std::make_unique<DrawSidesUsingPlanes>(vars,adj);
  }else
    sides = std::make_unique<DrawSidesUsingPoints>(vars,adj);
  caps = std::make_unique<DrawCaps>(adj);

}

VSSV::~VSSV(){}

void VSSV::drawSides(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  sides->draw(lightPosition,viewMatrix,projectionMatrix);
}

void VSSV::drawCaps(
    glm::vec4 const&lightPosition,
    glm::mat4 const&viewMatrix         ,
    glm::mat4 const&projectionMatrix   ){
  caps->draw(lightPosition,viewMatrix,projectionMatrix);
}

