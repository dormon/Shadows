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

Sintorn2::Sintorn2(vars::Vars& vars) : ShadowMethod(vars) {}

Sintorn2::~Sintorn2() {vars.erase("cssv.method");}



void Sintorn2::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  sintorn2::buildHierarchy(vars);

}
