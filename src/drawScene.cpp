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

  if(!vars.getBool("dontTimeGbuffer"))
  {
    ifExistStamp(vars, "gBuffer");
  }

  ifMethodExistCreateShadowMask(vars);

  doShading(vars);

  ifExistEndStamp(vars,"shading");

  if(vars.addOrGetBool("debug"))
    ifMethodExistsDrawDebug(vars);
}
