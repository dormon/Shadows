#pragma once

#include<CameraParam.h>
#include<BasicCamera/Fwd.h>
#include<SDL2CPP/MainLoop.h>
#include<Vars.h>

void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown);
