#include<glm/glm.hpp>

#include<TimeStamp.h>
#include<FunctionPrologue.h>

#include<CSSV/CSSV.h>
#include<CSSV/sides/draw.h>
#include<CSSV/caps/draw.h>


using namespace cssv;
using namespace ge::gl;
using namespace std;
using namespace glm;


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
  FUNCTION_CALLER();
  cssv::sides::draw(vars,lightPosition,viewMatrix,projectionMatrix);
}

void CSSV::drawCaps(
    vec4 const&lightPosition   ,
    mat4 const&viewMatrix      ,
    mat4 const&projectionMatrix){
  FUNCTION_CALLER();
  cssv::caps::draw(vars,lightPosition,viewMatrix,projectionMatrix);
}

