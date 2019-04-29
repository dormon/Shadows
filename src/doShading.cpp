#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <BasicCamera/Camera.h>
#include <Shading.h>
#include <FunctionPrologue.h>

void doShading(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  ge::gl::glDisable(GL_DEPTH_TEST);
  auto const cameraTransform            = vars.getReinterpret<basicCamera::CameraTransform >("cameraTransform" );
  auto       shading                    = vars.get<Shading>("shading");
  auto       lightPosition              = *vars.get<glm::vec4>("lightPosition");
  auto const cameraPositionInViewSpace  = glm::vec4(0, 0, 0, 1);
  auto const viewMatrix                 = cameraTransform->getView();
  auto const viewSpaceToWorldSpace      = glm::inverse(viewMatrix);
  auto       cameraPositionInWorldSpace = glm::vec3( viewSpaceToWorldSpace * cameraPositionInViewSpace);
  shading->draw(lightPosition,cameraPositionInWorldSpace,*vars.get<bool>("useShadows"));
}

