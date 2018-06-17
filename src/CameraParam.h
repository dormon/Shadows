#pragma once

#include <SDL2CPP/MainLoop.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/Camera.h>
#include <glm/glm.hpp>
#include <Vars.h>

struct CameraParam {
  std::string type            = "orbit";
  float       fovy            = glm::radians(90.f);
  float       near            = 0.1f;
  float       far             = 1000.f;
  float       sensitivity     = 0.01f;
  float       orbitZoomSpeed  = 0.2f;
  float       freeCameraSpeed = 1.f;
  CameraParam() {}
  CameraParam(std::shared_ptr<argumentViewer::ArgumentViewer> const& args);
};

std::shared_ptr<basicCamera::CameraProjection>createProjection(CameraParam const&p,glm::uvec2 const&windowSize);
std::shared_ptr<basicCamera::CameraTransform>createView(CameraParam const&p);

void mouseMoveCamera(
    std::shared_ptr<basicCamera::CameraTransform>const&cameraTransform,
    SDL_Event const&event,
    CameraParam const&p,
    glm::uvec2 const&windowSize);

void printCameraPosition(std::shared_ptr<basicCamera::CameraTransform>const&cameraTransform);
