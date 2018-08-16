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
find_package(geGL 1.0.5 CONFIG REQUIRED)
find_package(SDL2CPP 1.0 CONFIG REQUIRED)
find_package(BasicCamera 1.0 CONFIG REQUIRED)

find_package(assimp CONFIG REQUIRED)
if (assimp_FOUND)
  set(ASSIMP_LIBRARY "assimp")
  add_library(${ASSIMP_LIBRARY} SHARED IMPORTED)
  set_target_properties(${ASSIMP_LIBRARY} PROPERTIES IMPORTED_LOCATION "${ASSIMP_LIBRARY_DIRS}/libassimp.so")
endif(assimp_FOUND)


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
  src/FastAdjacency.cpp
  src/TimeStamp.cpp
  src/StencilBufferToShadowMaskProgram.cpp
  src/CSSV/CSSV.cpp
  src/CSSV/ExtractSilhouettes.cpp
  src/CSSV/BasicExtractSilhouettes.cpp
  src/CSSV/PlanesExtractSilhouettes.cpp
  src/CSSV/InterleavedPlanesExtractSilhouettes.cpp
  src/CSSV/Param.cpp
  src/CSSV/DrawSidesProgram.cpp
  src/CSSV/DrawCapsProgram.cpp
  src/CSSV/DrawCaps.cpp
  src/CSSV/DrawSides.cpp
  src/CubeShadowMapping/CubeShadowMapping.cpp
  src/CubeShadowMapping/Params.cpp
  src/VSSV/VSSV.cpp
  src/VSSV/Params.cpp
  src/VSSV/DrawCaps.cpp
  src/CSSVSOE.cpp
  src/Sintorn.cpp
  src/SintornTiles.cpp
  src/RSSV.cpp
  src/RSSVTiles.cpp
  src/DrawPrimitive.cpp
  src/ShadowVolumes.cpp
  src/util.cpp
  src/mainUtil.cpp
  src/CameraParam.cpp
  src/TestParam.cpp
  src/SintornParam.cpp
  src/RSSVParam.cpp
  src/Function.cpp
  src/FunctionImpl.cpp
  src/createGBuffer.cpp
  src/imguiVars.cpp
  src/Barrier.cpp
  )

set(INCLUDES
  src/Application.h
  src/CameraPath.h
  src/CSV.h
  src/Deferred.h
  src/Model.h
  src/Shading.h
  src/ShadowMethod.h
  src/FastAdjacency.h
  src/TimeStamp.h
  src/StencilBufferToShadowMaskProgram.h
  src/CSSV/Fwd.h
  src/CSSV/CSSV.h
  src/CSSV/DrawSidesShaders.h
  src/CSSV/DrawCapsShaders.h
  src/CSSV/ExtractSilhouetteShader.h
  src/CSSV/ExtractSilhouettes.h
  src/CSSV/BasicExtractSilhouettes.h
  src/CSSV/PlanesExtractSilhouettes.h
  src/CSSV/InterleavedPlanesExtractSilhouettes.h
  src/CSSV/InterleavedPlanesShader.h
  src/CSSV/DrawSidesProgram.h
  src/CSSV/DrawCapsProgram.h
  src/CSSV/DrawCaps.h
  src/CSSV/DrawSides.h
  src/CSSV/Param.h
  src/CubeShadowMapping/CubeShadowMapping.h
  src/CubeShadowMapping/ShadowMapToShadowMaskShaders.h
  src/CubeShadowMapping/CreateShadowMapShaders.h
  src/CubeShadowMapping/Params.h
  src/VSSV/VSSV.h
  src/VSSV/Shaders.h
  src/VSSV/Params.h
  src/VSSV/DrawCaps.h
  src/CSSVSOE.h
  src/Sintorn.h
  src/SintornTiles.h
  src/SintornShaders.h
  src/RSSV.h
  src/RSSVTiles.h
  src/RSSVShaders.h
  src/DrawPrimitive.h
  src/ProgramExtension.h
  src/ShadowVolumes.h
  src/ShadowVolumesShaders.h
  src/SilhouetteShaders.h
  src/util.h
  src/mainUtil.h
  src/CameraParam.h
  src/TestParam.h
  src/SintornParam.h
  src/RSSVParam.h
  src/Function.h
  src/FunctionImpl.h
  src/createGBuffer.h
  src/imguiVars.h
  src/Barrier.h
  src/Simplex.h
  )

add_executable(${APP_NAME} ${SOURCES} ${INCLUDES})

target_include_directories(${APP_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${APP_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${APP_NAME} PUBLIC SDL2CPP::SDL2CPP BasicCamera::BasicCamera geGL::geGL ArgumentViewer::ArgumentViewer ${ASSIMP_LIBRARIES} imguiSDL2OpenGL::imguiSDL2OpenGL Vars::Vars)

