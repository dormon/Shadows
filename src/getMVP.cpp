#include <Vars/Vars.h>
#include <BasicCamera/Camera.h>

glm::mat4 getMVP(vars::Vars&vars){
  auto const cameraProjection = vars.getReinterpret<basicCamera::CameraProjection>("cameraProjection");
  auto const cameraTransform  = vars.getReinterpret<basicCamera::CameraTransform >("cameraTransform" );
  return cameraProjection->getProjection() * cameraTransform->getView();
}
