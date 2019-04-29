#include <limits>
#include <Vars/Vars.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadCameraParams(
    vars::Vars&                                            vars,
    std::shared_ptr<argumentViewer::ArgumentViewer> const& args) {
  vars.addString("args.camera.type")            =      args->gets("--camera-type", "free", "orbit/free camera type");
  vars.addFloat ("args.camera.fovy")            =      args->getf32("--camera-fovy", 1.5707963267948966f,                   "camera field of view in y direction");
  vars.addFloat ("args.camera.near")            =      args->getf32("--camera-near", 0.1f, "camera near plane position");
  vars.addFloat ("args.camera.far")             =      args->getf32("--camera-far", std::numeric_limits<float>::infinity(),                   "camera far plane position");
  vars.addFloat ("args.camera.sensitivity")     =      args->getf32("--camera-sensitivity", 0.01f, "camera sensitivity");
  vars.addFloat ("args.camera.orbitZoomSpeed")  =      args->getf32("--camera-zoomSpeed", 0.2f, "orbit camera zoom speed");
  vars.addFloat ("args.camera.freeCameraSpeed") =      args->getf32("--camera-speed", 1.f, "free camera speed");
}

