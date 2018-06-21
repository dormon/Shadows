#pragma once

#include<CameraParam.h>
#include<BasicCamera/Fwd.h>
#include<SDL2CPP/MainLoop.h>
#include<Vars.h>
#include<ArgumentViewer/Fwd.h>

void loadBasicApplicationParameters(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&args);

void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown);


void writeCSVHeaderIfFirstLine(
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement);

void writeMeasurementIntoCSV(
    vars::Vars&vars,
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement,
    size_t idOfMeasurement);

class CameraPath;
void setCameraAccordingToKeyFrame(std::shared_ptr<CameraPath>const&cameraPath,vars::Vars&vars,size_t keyFrame);
