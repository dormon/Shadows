#include<mainUtil.h>
#include<BasicCamera/FreeLookCamera.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include<util.h>
#include<TxtUtils/TxtUtils.h>
#include<CameraPath.h>
#include<geGL/StaticCalls.h>
#include<ShadowMethod.h>
#include<Shading.h>
#include<sstream>
#include<Methods.h>
#include<Vars/Caller.h>



void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown) {
  vars::Caller caller(vars,__FUNCTION__);
  auto const type = vars.getString("args.camera.type");
  auto const freeCameraSpeed = vars.getFloat("args.camera.freeCameraSpeed");
  if (type != "free") return;
  auto const freeLook = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  for (int a = 0; a < 3; ++a)
    freeLook->move(a, float(keyDown["d s"[a]] - keyDown["acw"[a]]) *
                          freeCameraSpeed);
} 

void updateLightPosViewUpFromCamera(vars::Vars& vars)
{
	if (vars.getString("args.camera.type") == "free")
	{
		basicCamera::FreeLookCamera* const freeLookCam = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");

		glm::vec3 pos = freeLookCam->getPosition();
		glm::mat4 rv = freeLookCam->getRotation();
		glm::vec3 view = glm::normalize(-glm::vec3(glm::row(rv, 2)));
		glm::vec3 up = glm::normalize(glm::vec3(glm::row(rv, 1)));
		
		std::cout << "\n";
		std::cout << "Light pos: " << pos.x << " " << pos.y << " " << pos.z << std::endl;
		std::cout << "Light dir: " << view.x << " awd" << view.y << " " << view.z << std::endl;
		std::cout << "Light up: " << up.x << " " << up.y << " " << up.z << std::endl;

		*vars.get<glm::vec3>("lightUp") = up;
		*vars.get<glm::vec3>("lightView") = view;
		*vars.get<glm::vec4>("lightPosition") = glm::vec4(pos, 1);

		vars.updateTicks("lightUp");
		vars.updateTicks("lightView");
		vars.updateTicks("lightPosition");
	}
}

