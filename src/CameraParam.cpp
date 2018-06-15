#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CameraParam.h>

CameraParam::CameraParam(
    std::shared_ptr<argumentViewer::ArgumentViewer> const& arg) {
  type = arg->gets("--camera-type", "free", "orbit/free camera type");
  fovy = arg->getf32("--camera-fovy", 1.5707963267948966f,
                     "camera field of view in y direction");
  near = arg->getf32("--camera-near", 0.1f, "camera near plane position");
  far  = arg->getf32("--camera-far", std::numeric_limits<float>::infinity(),
                    "camera far plane position");
  sensitivity =
      arg->getf32("--camera-sensitivity", 0.01f, "camera sensitivity");
  orbitZoomSpeed =
      arg->getf32("--camera-zoomSpeed", 0.2f, "orbit camera zoom speed");
  freeCameraSpeed = arg->getf32("--camera-speed", 1.f, "free camera speed");
}

std::shared_ptr<basicCamera::CameraProjection> createProjection(
    CameraParam const& p, glm::uvec2 const& windowSize) {
  return std::make_shared<basicCamera::PerspectiveCamera>(
      p.fovy, (float)windowSize.x / (float)windowSize.y, p.near, p.far);
}

std::shared_ptr<basicCamera::CameraTransform>createView(CameraParam const&p){
  if (p.type == "orbit")
    return std::make_shared<basicCamera::OrbitCamera>();
  else if (p.type == "free")
    return std::make_shared<basicCamera::FreeLookCamera>();
  else {
    std::cerr << "ERROR: --camera-type is incorrect" << std::endl;
    exit(0);
  }
}

void mouseMoveFreeLookCamera(
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform,
    SDL_Event const&                                     event,
    CameraParam const&                                   p) {
  auto freeCamera =
      std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
  if (event.motion.state & SDL_BUTTON_LMASK) {
    freeCamera->setAngle(
        1, freeCamera->getAngle(1) + float(event.motion.xrel) * p.sensitivity);
    freeCamera->setAngle(
        0, freeCamera->getAngle(0) + float(event.motion.yrel) * p.sensitivity);
  }
}

void mouseMoveOrbitCamera(
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform,
    SDL_Event const&                                     event,
    CameraParam const&                                   p,
    glm::uvec2 const&                                    windowSize) {
  if (event.motion.state & SDL_BUTTON_LMASK) {
    auto orbitCamera =
        std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
    if (orbitCamera) {
      orbitCamera->addXAngle(float(event.motion.yrel) * p.sensitivity);
      orbitCamera->addYAngle(float(event.motion.xrel) * p.sensitivity);
    }
  }
  if (event.motion.state & SDL_BUTTON_RMASK) {
    auto orbitCamera =
        std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
    if (orbitCamera)
      orbitCamera->addDistance(float(event.motion.yrel) * p.orbitZoomSpeed);
  }
  if (event.motion.state & SDL_BUTTON_MMASK) {
    auto orbitCamera =
        std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
    orbitCamera->addXPosition(+orbitCamera->getDistance() *
                              float(event.motion.xrel) / float(windowSize.x) *
                              2.f);
    orbitCamera->addYPosition(-orbitCamera->getDistance() *
                              float(event.motion.yrel) / float(windowSize.y) *
                              2.f);
  }
}

void mouseMoveCamera(
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform,
    SDL_Event const&                                     event,
    CameraParam const&                                   p,
    glm::uvec2 const&                                    windowSize) {
  if (p.type == "orbit")
    mouseMoveOrbitCamera(cameraTransform, event, p, windowSize);
  if (p.type == "free") mouseMoveFreeLookCamera(cameraTransform, event, p);
}

void printCameraPosition(
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform) {
  auto flc =
      std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
  if (!flc) return;
  auto rv   = flc->getRotation();
  auto pos  = flc->getPosition();
  auto up   = glm::normalize(glm::vec3(glm::row(rv, 1)));
  auto view = glm::normalize(-glm::vec3(glm::row(rv, 2)));
  std::cout << pos.x << "," << pos.y << "," << pos.z << ",";
  std::cout << view.x << "," << view.y << "," << view.z << ",";
  std::cout << up.x << "," << up.y << "," << up.z << std::endl;
}
