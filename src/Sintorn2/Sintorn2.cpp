#include <Sintorn2/Sintorn2.h>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <startStop.h>
#include <sstream>
#include <algorithm>
#include <BallotShader.h>

#include <Sintorn2/buildHierarchy.h>
#include <Sintorn2/debug/drawDebug.h>

Sintorn2::Sintorn2(vars::Vars& vars) : ShadowMethod(vars) {}

Sintorn2::~Sintorn2() {vars.erase("cssv.method");}



void Sintorn2::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  FUNCTION_CALLER();
  //glFinish();
  ifExistStamp("");
  sintorn2::buildHierarchy(vars);
  ifExistStamp("buildHierarchy");

}

void Sintorn2::drawDebug(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  FUNCTION_CALLER();
  *vars.addOrGet<glm::vec4>("sintorn2.method.debug.lightPosition"   ) = lightPosition   ;
  *vars.addOrGet<glm::mat4>("sintorn2.method.debug.viewMatrix"      ) = viewMatrix      ;
  *vars.addOrGet<glm::mat4>("sintorn2.method.debug.projectionMatrix") = projectionMatrix;
  sintorn2::drawDebug(vars);
}
