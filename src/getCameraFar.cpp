#include <Vars/Vars.h>
#include <BasicCamera/PerspectiveCamera.h>

float getCameraFar(vars::Vars&vars){
  auto const proj = vars.getReinterpret<basicCamera::PerspectiveCamera>("cameraProjection");
  return proj->getFar();
}
