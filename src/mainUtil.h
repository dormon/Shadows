#pragma once

#include<CameraParam.h>
#include<BasicCamera/Fwd.h>
#include<SDL2CPP/MainLoop.h>

void moveCameraWSAD(
    CameraParam const&                                   cameraParam,
    std::shared_ptr<basicCamera::CameraTransform> const& cameraTransform,
    std::map<SDL_Keycode, bool>                          keyDown);
