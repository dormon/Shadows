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


