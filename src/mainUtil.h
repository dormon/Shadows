#pragma once

#include<SDL2CPP/MainLoop.h>
#include<Vars/Vars.h>

void moveCameraWSAD(
    vars::Vars&vars,
    std::map<SDL_Keycode, bool>                          keyDown);

void createGBuffer(vars::Vars&vars);

void ifMethodExistCreateShadowMask(vars::Vars&vars);

void doShading(vars::Vars&vars);
