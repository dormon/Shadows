cmake_minimum_required(VERSION 3.10.1)

project(Shadows)

SET(CMAKE_CXX_STANDARD 14)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(REVISION_VERSION 0)

#include(GenerateExportHeader)

set(APP_NAME Shadows)

find_package(ArgumentViewer 1.0 CONFIG REQUIRED)
find_package(geGL 1.0.4 CONFIG REQUIRED)
find_package(SDL2CPP 1.0 CONFIG REQUIRED)
find_package(BasicCamera 1.0 CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imguiSDL2OpenGL CONFIG REQUIRED)
find_package(Vars CONFIG REQUIRED)
#find_package(SDL2 CONFIG REQUIRED)

set(SOURCES
  src/Application.cpp
  src/main.cpp
  src/Model.cpp
  src/CameraPath.cpp
  src/CSV.cpp
  src/Shading.cpp
  src/ShadowMethod.cpp
  src/CubeShadowMapping.cpp
  src/FastAdjacency.cpp
  src/TimeStamp.cpp
  src/CSSV.cpp
  src/CSSVExtractSilhouettes.cpp
  src/CSSVBasicExtractSilhouettes.cpp
  src/CSSVPlanesExtractSilhouettes.cpp
  src/CSSVInterleavedPlanesExtractSilhouettes.cpp
  src/CSSVSOE.cpp
  src/Sintorn.cpp
  src/SintornTiles.cpp
  src/RSSV.cpp
  src/RSSVTiles.cpp
  src/VSSV.cpp
  src/DrawPrimitive.cpp
  src/ShadowVolumes.cpp
  src/util.cpp
  src/mainUtil.cpp
  src/CameraParam.cpp
  src/TestParam.cpp
  src/CubeSMParam.cpp
  src/CSSVParam.cpp
  src/VSSVParam.cpp
  src/SintornParam.cpp
  src/RSSVParam.cpp
  src/Function.cpp
  src/FunctionImpl.cpp
  src/createGBuffer.cpp
  src/imguiVars.cpp
  )

set(INCLUDES
  src/Application.h
  src/CameraPath.h
  src/CSV.h
  src/Deferred.h
  src/Model.h
  src/Shading.h
  src/ShadowMethod.h
  src/CubeShadowMapping.h
  src/CubeShadowMappingShaders.h
  src/FastAdjacency.h
  src/TimeStamp.h
  src/CSSV.h
  src/CSSVShaders.h
  src/CSSVSOE.h
  src/CSSVExtractSilhouetteShader.h
  src/CSSVExtractSilhouettes.h
  src/CSSVBasicExtractSilhouettes.h
  src/CSSVPlanesExtractSilhouettes.h
  src/CSSVInterleavedPlanesExtractSilhouettes.h
  src/Sintorn.h
  src/SintornTiles.h
  src/SintornShaders.h
  src/RSSV.h
  src/RSSVTiles.h
  src/RSSVShaders.h
  src/VSSV.h
  src/VSSVShaders.h
  src/DrawPrimitive.h
  src/ProgramExtension.h
  src/ShadowVolumes.h
  src/ShadowVolumesShaders.h
  src/SilhouetteShaders.h
  src/util.h
  src/mainUtil.h
  src/CameraParam.h
  src/TestParam.h
  src/CubeSMParam.h
  src/CSSVParam.h
  src/VSSVParam.h
  src/SintornParam.h
  src/RSSVParam.h
  src/Function.h
  src/FunctionImpl.h
  src/createGBuffer.h
  src/imguiVars.h
  )

add_executable(${APP_NAME} ${SOURCES} ${INCLUDES})

target_include_directories(${APP_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${APP_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${APP_NAME} PUBLIC SDL2CPP::SDL2CPP BasicCamera::BasicCamera geGL::geGL ArgumentViewer::ArgumentViewer assimp imguiSDL2OpenGL::imguiSDL2OpenGL Vars::Vars)

