#include <Vars/Vars.h>
#include <BasicCamera/Camera.h>
#include <FunctionPrologue.h>
#include <ShadowMethod.h>

void ifMethodExistsDrawDebug(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  if (!vars.has("shadowMethod"))return;
  auto const cameraProjection = vars.getReinterpret<basicCamera::CameraProjection>("cameraProjection");
  auto const cameraTransform  = vars.getReinterpret<basicCamera::CameraTransform >("cameraTransform" );
  auto       method           = vars.getReinterpret<ShadowMethod>("shadowMethod");
  auto const lightPosition    = *vars.get<glm::vec4>("lightPosition");
  method->drawDebug(lightPosition,cameraTransform->getView(),cameraProjection->getProjection());
}

