#include <iostream>
#include <SDL2CPP/MainLoop.h>
#include <SDL2CPP/Window.h>
#include <Vars/Vars.h>
#include <BasicCamera/FreeLookCamera.h>
#include <TxtUtils/TxtUtils.h>
#include <FunctionPrologue.h>
#include <CameraPath.h>
#include <TimeStamp.h>
#include <drawScene.h>
#include <CSV.h>

void setCameraAccordingToKeyFrame(std::shared_ptr<CameraPath>const&cameraPath,vars::Vars&vars,size_t keyFrame){
  vars::Caller caller(vars,__FUNCTION__);
  auto keypoint =
      cameraPath->getKeypoint(float(keyFrame) / float(vars.getSizeT("test.flyLength")));
  auto flc = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
  flc->setPosition(keypoint.position);
  flc->setRotation(keypoint.viewVector, keypoint.upVector);
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
  vars::Caller caller(vars,__FUNCTION__);
  std::vector<std::string> line;
  line.push_back(txtUtils::valueToString(idOfMeasurement));
  for (auto const& x : measurement)
    if (x.first != "")
      line.push_back(txtUtils::valueToString(
          x.second / float(vars.getSizeT("test.framesPerMeasurement"))));
  csv.push_back(line);
}



void measureFly(vars::Vars&vars){
  FUNCTION_CALLER();

  auto mainLoop = *vars.get<sdl2cpp::MainLoop*>("mainLoop");
  auto window   = *vars.get<sdl2cpp::Window  *>("window"  );
  if (vars.getString("test.flyKeyFileName") == "") {
    std::cerr << "camera path file is empty" << std::endl;
    mainLoop->removeWindow(window->getId());
    return;
  }
  auto cameraPath =
      std::make_shared<CameraPath>(false, vars.getString("test.flyKeyFileName"));
  std::map<std::string, float> measurement;
  vars.get<TimeStamp>("timeStamp")->setPrinter([&](std::vector<std::string> const& names,
                              std::vector<float> const&       values) {
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] != "") {
        if (measurement.count(names[i]) == 0) measurement[names[i]] = 0.f;
        measurement[names[i]] += values[i];
      }
  });

  std::vector<std::vector<std::string>> csv;
  for (size_t k = 0; k < vars.getSizeT("test.flyLength"); ++k) {
    setCameraAccordingToKeyFrame(cameraPath,vars,k);

    for (size_t f = 0; f < vars.getSizeT("test.framesPerMeasurement"); ++f) drawScene(vars);

    writeCSVHeaderIfFirstLine(csv,measurement);
    writeMeasurementIntoCSV(vars,csv,measurement,k);

    measurement.clear();
    window->swap();
  }
  std::string output = vars.getString("test.outputName") + ".csv";
  saveCSV(output, csv);
  mainLoop->removeWindow(window->getId());
}
