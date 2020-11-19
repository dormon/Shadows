#pragma once

#include<SDL2CPP/MainLoop.h>
#include<Vars/Fwd.h>

void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown);

void updateLightPosViewUpFromCamera(vars::Vars& vars);

void createGBuffer(vars::Vars&vars);

void ifMethodExistCreateShadowMask(vars::Vars&vars);

void doShading(vars::Vars&vars);
