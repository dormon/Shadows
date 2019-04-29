#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CameraParam.h>

#include<Barrier.h>

void createProjection(vars::Vars& vars) {
  if(notChanged(vars,"all",__FUNCTION__,{"args.camera.fovy","args.camera.near","args.camera.far","windowSize"}))return;

  auto const fovy       = vars.getFloat("args.camera.fovy");
  auto const near       = vars.getFloat("args.camera.near");
  auto const far        = vars.getFloat("args.camera.far");
  auto const windowSize = glm::vec2(*vars.get<glm::uvec2>("windowSize"));
  auto const aspect     = windowSize.x / windowSize.y;
  vars.reCreate<basicCamera::PerspectiveCamera>("cameraProjection", fovy, aspect,
                                           near, far);
}

void createView(vars::Vars& vars) {
  auto const type = vars.getString("args.camera.type");
  if (type == "orbit")
    vars.add<basicCamera::OrbitCamera>("cameraTransform");
  else if (type == "free")
    vars.add<basicCamera::FreeLookCamera>("cameraTransform");
  else {
    std::cerr << "ERROR: --camera-type is incorrect" << std::endl;
    exit(0);
  }
}

void mouseMoveFreeLookCamera(vars::Vars& vars, SDL_Event const& event) {
  auto freeCamera =
      vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  auto const sensitivity = vars.getFloat("args.camera.sensitivity");
  auto const xrel           = static_cast<float>(event.motion.xrel);
  auto const yrel           = static_cast<float>(event.motion.yrel);
  if (event.motion.state & SDL_BUTTON_LMASK) {
    freeCamera->setAngle(
        1, freeCamera->getAngle(1) + xrel * sensitivity);
    freeCamera->setAngle(
        0, freeCamera->getAngle(0) + yrel * sensitivity);
  }
}

void mouseMoveOrbitCamera(vars::Vars& vars, SDL_Event const& event) {
  auto orbitCamera =
      vars.getReinterpret<basicCamera::OrbitCamera>("cameraTransform");
  auto const windowSize     = vars.get<glm::uvec2>("windowSize");
  auto const sensitivity    = vars.getFloat("args.camera.sensitivity");
  auto const orbitZoomSpeed = vars.getFloat("args.camera.orbitZoomSpeed");
  auto const xrel           = static_cast<float>(event.motion.xrel);
  auto const yrel           = static_cast<float>(event.motion.yrel);
  auto const mState         = event.motion.state;
  if (mState & SDL_BUTTON_LMASK) {
    if (orbitCamera) {
      orbitCamera->addXAngle(yrel * sensitivity);
      orbitCamera->addYAngle(xrel * sensitivity);
    }
  }
  if (mState & SDL_BUTTON_RMASK) {
    if (orbitCamera) orbitCamera->addDistance(yrel * orbitZoomSpeed);
  }
  if (mState & SDL_BUTTON_MMASK) {
    orbitCamera->addXPosition(+orbitCamera->getDistance() * xrel /
                              float(windowSize->x) * 2.f);
    orbitCamera->addYPosition(-orbitCamera->getDistance() * yrel /
                              float(windowSize->y) * 2.f);
  }
}

void mouseMoveCamera(vars::Vars& vars, SDL_Event const& event) {
  auto const type = vars.getString("args.camera.type");
  if (type == "orbit") mouseMoveOrbitCamera(vars, event);
  if (type == "free") mouseMoveFreeLookCamera(vars, event);
}

void printCameraPosition(vars::Vars&vars) {
  auto flc = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  if (!flc) return;
  auto rv   = flc->getRotation();
  auto pos  = flc->getPosition();
  auto up   = glm::normalize(glm::vec3(glm::row(rv, 1)));
  auto view = glm::normalize(-glm::vec3(glm::row(rv, 2)));
  std::cout << pos.x << "," << pos.y << "," << pos.z << ",";
  std::cout << view.x << "," << view.y << "," << view.z << ",";
  std::cout << up.x << "," << up.y << "," << up.z << std::endl;
}
