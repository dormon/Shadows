#pragma once

#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/Camera.h>
#include <SDL2CPP/MainLoop.h>
#include <Vars.h>
#include <glm/glm.hpp>

void loadCameraParams(
    vars::Vars&                                            vars,
    std::shared_ptr<argumentViewer::ArgumentViewer> const& args);

void createView(vars::Vars& vars);
void createProjection(vars::Vars& vars);

void mouseMoveCamera(vars::Vars& vars, SDL_Event const& event);

void printCameraPosition(vars::Vars& vars);
