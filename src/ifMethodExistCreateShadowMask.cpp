#include <Vars/Vars.h>
#include <BasicCamera/Camera.h>
#include <FunctionPrologue.h>
#include <ShadowMethod.h>

void ifMethodExistCreateShadowMask(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  if (!vars.has("shadowMethod"))return;
  auto const cameraProjection = vars.getReinterpret<basicCamera::CameraProjection>("cameraProjection");
  auto const cameraTransform  = vars.getReinterpret<basicCamera::CameraTransform >("cameraTransform" );
  auto       method           = vars.getReinterpret<ShadowMethod>("shadowMethod");
  auto const lightPosition    = *vars.get<glm::vec4>("lightPosition");
  vars.get<ge::gl::Texture>("shadowMask")->clear(0,GL_RED,GL_FLOAT);
  method->create(lightPosition,cameraTransform->getView(),cameraProjection->getProjection());
}

