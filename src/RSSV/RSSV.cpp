#include <RSSV/RSSV.h>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <startStop.h>
#include <sstream>
#include <algorithm>
#include <BallotShader.h>

#include <RSSV/buildHierarchy.h>
#include <RSSV/computeShadowFrusta.h>
#include <RSSV/rasterize.h>
#include <RSSV/traverse.h>
#include <RSSV/merge.h>
#include <RSSV/debug/drawDebug.h>
#include <RSSV/extractSilhouettes.h>

namespace rssv{

RSSV::RSSV(vars::Vars& vars) : ShadowMethod(vars) {}

RSSV::~RSSV() {vars.erase("cssv.method");}



void RSSV::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  FUNCTION_CALLER();
  *vars.addOrGet<glm::vec4>("rssv.method.lightPosition"   ) = lightPosition   ;
  *vars.addOrGet<glm::mat4>("rssv.method.viewMatrix"      ) = viewMatrix      ;
  *vars.addOrGet<glm::mat4>("rssv.method.projectionMatrix") = projectionMatrix;

  //return;
  //glFinish();
  ifExistStamp("");
  rssv::computeShadowFrusta(vars);
  ifExistStamp("computeShadowFrusta");
  rssv::extractSilhouettes(vars);
  ifExistStamp("extractSilhouettes");
  rssv::buildHierarchy(vars);
  ifExistStamp("buildHierarchy");
  rssv::traverse(vars);
  ifExistStamp("traverse");
  //rssv::rasterize(vars);
  //ifExistStamp("rasterize");
  rssv::merge(vars);
  ifExistStamp("merge");

}

void RSSV::drawDebug(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  FUNCTION_CALLER();
  *vars.addOrGet<glm::vec4>("rssv.method.debug.lightPosition"   ) = lightPosition   ;
  *vars.addOrGet<glm::mat4>("rssv.method.debug.viewMatrix"      ) = viewMatrix      ;
  *vars.addOrGet<glm::mat4>("rssv.method.debug.projectionMatrix") = projectionMatrix;
  rssv::drawDebug(vars);
}

}
