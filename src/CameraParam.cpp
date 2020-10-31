#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CameraParam.h>
#include <Vars/Vars.h>
#include <Barrier.h>

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

#include <fstream>

void storeCamera(vars::Vars&vars){
  if(!vars.getBool("args.camera.remember"))return;
  std::string name = "storedCamera.txt";
  auto&cam = *vars.get<basicCamera::FreeLookCamera>("cameraTransform");

  auto pos = cam.getPosition();
  auto a0 = cam.getAngle(0);
  auto a1 = cam.getAngle(1);
  auto a2 = cam.getAngle(2);

  std::ofstream file (name);
  if(!file.is_open()){
    std::cerr << "cannot open camera file: " << name << std::endl;
    return;
  }

  file << pos.x << std::endl;
  file << pos.y << std::endl;
  file << pos.z << std::endl;
  file << a0 << std::endl;
  file << a1 << std::endl;
  file << a2 << std::endl;

  file.close();

}

void loadCamera(vars::Vars&vars){
  if(!vars.getBool("args.camera.remember"))return;
  std::string name = "storedCamera.txt";
  auto&cam = *vars.get<basicCamera::FreeLookCamera>("cameraTransform");

  std::ifstream file (name);
  if(!file.is_open()){
    std::cerr << "cannot open camera file: " << name << std::endl;
    return;
  }
  glm::vec3 pos;
  float data[6];
  file >> data[0];
  file >> data[1];
  file >> data[2];
  file >> data[3];
  file >> data[4];
  file >> data[5];

  pos.x = *(data+0);
  pos.y = *(data+1);
  pos.z = *(data+2);
  float a0,a1,a2;
  a0 = *(data+3);
  a1 = *(data+4);
  a2 = *(data+5);

  cam.setPosition(pos);
  cam.setAngle(0,a0);
  cam.setAngle(1,a1);
  cam.setAngle(2,a2);

  file.close();
}

void createView(vars::Vars& vars) {
  auto const type = vars.getString("args.camera.type");
  if (type == "orbit")
    vars.add<basicCamera::OrbitCamera>("cameraTransform");
  else if (type == "free"){
    vars.add<basicCamera::FreeLookCamera>("cameraTransform");
    loadCamera(vars);
  }else {
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
