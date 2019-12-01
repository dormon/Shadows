#include<Vars/Vars.h>
#include <renderModelToGBuffer.h>
#include <ifExistBeginStamp.h>
#include <ifExistStamp.h>
#include <ifExistEndStamp.h>
#include <doShading.h>
#include <ifMethodExistCreateShadowMask.h>
#include <ifMethodExistsDrawDebug.h>

void drawScene(vars::Vars&vars){
  //createMethod(vars);

  ifExistBeginStamp(vars);

  renderModelToGBuffer(vars);

  ifExistStamp(vars,"gBuffer");

  ifMethodExistCreateShadowMask(vars);

  doShading(vars);

  ifExistEndStamp(vars,"shading");

  ifMethodExistsDrawDebug(vars);
}
