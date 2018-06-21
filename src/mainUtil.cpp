#include<mainUtil.h>
#include<BasicCamera/FreeLookCamera.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include<util.h>
#include<TxtUtils/TxtUtils.h>
#include<CameraPath.h>

void loadBasicApplicationParameters(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&args){
  *vars.add<glm::uvec2 >("windowSize"     ) = vector2uvec2(args->getu32v("--window-size", {512, 512}, "window size"));
  *vars.add<glm::vec4  >("lightPosition"  ) = vector2vec4(args->getf32v("--light", {0.f, 1000.f, 0.f, 1.f}, "light position"));
  vars.addString        ("modelName"      ) = args->gets("--model", "/media/windata/ft/prace/models/2tri/2tri.3ds","model file name");
  vars.addBool          ("useShadows"     ) = !args->isPresent("--no-shadows", "turns off shadows");
  vars.addBool          ("verbose"        ) = args->isPresent("--verbose", "toggle verbose mode");
  vars.addString        ("methodName"     ) = args->gets("--method", "","name of shadow method: ""cubeShadowMapping/cssv/sintorn/rssv/vssv/cssvsoe");
  vars.addSizeT         ("wavefrontSize"  ) = args->getu32("--wavefrontSize", 0,"warp/wavefront size, usually 32 for NVidia and 64 for AMD");
  vars.addSizeT         ("maxMultiplicity") = args->getu32("--maxMultiplicity", 2,"max number of triangles that share the same edge");
  vars.addBool          ("zfail"          ) = args->getu32("--zfail", 1, "shadow volumes zfail 0/1");
}

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

void writeCSVHeaderIfFirstLine(
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement){
  if (csv.size() != 0) return;
  std::vector<std::string>line;
  line.push_back("frame");
  for (auto const& x : measurement)
    if (x.first != "") line.push_back(x.first);
  csv.push_back(line);
}

void writeMeasurementIntoCSV(
    vars::Vars&vars,
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement,
    size_t idOfMeasurement){
  std::vector<std::string> line;
  line.push_back(txtUtils::valueToString(idOfMeasurement));
  for (auto const& x : measurement)
    if (x.first != "")
      line.push_back(txtUtils::valueToString(
          x.second / float(vars.getSizeT("test.framesPerMeasurement"))));
  csv.push_back(line);
}

void setCameraAccordingToKeyFrame(std::shared_ptr<CameraPath>const&cameraPath,vars::Vars&vars,size_t keyFrame){
  auto keypoint =
      cameraPath->getKeypoint(float(keyFrame) / float(vars.getSizeT("test.flyLength")));
  auto flc = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  flc->setPosition(keypoint.position);
  flc->setRotation(keypoint.viewVector, keypoint.upVector);
}

