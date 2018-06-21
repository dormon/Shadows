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
