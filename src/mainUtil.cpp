#include<mainUtil.h>
#include<BasicCamera/FreeLookCamera.h>

void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown) {
  auto const type = vars.getString("camera.type");
  auto const freeCameraSpeed = vars.getFloat("camera.freeCameraSpeed");
  if (type != "free") return;
  auto const freeLook = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  for (int a = 0; a < 3; ++a)
    freeLook->move(a, float(keyDown["d s"[a]] - keyDown["acw"[a]]) *
                          freeCameraSpeed);
}

