#include<mainUtil.h>
#include<BasicCamera/FreeLookCamera.h>

void moveCameraWSAD(
    CameraParam const&                                   cameraParam,
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform,
    std::map<SDL_Keycode, bool>                          keyDown) {
  if (cameraParam.type != "free") return;
  auto freeLook =
      std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
  for (int a = 0; a < 3; ++a)
    freeLook->move(a, float(keyDown["d s"[a]] - keyDown["acw"[a]]) *
                          cameraParam.freeCameraSpeed);
}

