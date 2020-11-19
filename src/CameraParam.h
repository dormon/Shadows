#pragma once

#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/Camera.h>
#include <SDL2CPP/MainLoop.h>
#include <Vars/Fwd.h>
#include <glm/glm.hpp>


void createView(vars::Vars& vars);
void createProjection(vars::Vars& vars);

void mouseMoveCamera(vars::Vars& vars, SDL_Event const& event);

void printCameraPosition(vars::Vars& vars);
void storeCamera(vars::Vars&vars);
void loadCamera(vars::Vars&vars);
