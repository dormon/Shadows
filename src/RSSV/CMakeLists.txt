cmake_minimum_required(VERSION 3.12.1)

SET(CMAKE_CXX_STANDARD 17)

project(RSSV)

add_library(${PROJECT_NAME} STATIC
  RSSV.cpp
  RSSV.h
  allocateHierarchy.cpp
  allocateHierarchy.h
  mortonShader.cpp
  mortonShader.h
  param.cpp
  param.h
  createBuildHierarchyProgram.cpp
  createBuildHierarchyProgram.h
  buildHierarchy.cpp
  buildHierarchy.h
  buildHierarchyShader.cpp
  buildHierarchyShader.h
  reduceShader.cpp
  reduceShader.h
  depthToZShader.cpp
  depthToZShader.h
  quantizeZShader.cpp
  quantizeZShader.h
  config.h
  propagateAABBShader.cpp
  propagateAABBShader.h
  propagateAABB.cpp
  propagateAABB.h
  createPropagateAABBProgram.cpp
  createPropagateAABBProgram.h
  computeConfig.cpp
  computeConfig.h
  configShader.cpp
  configShader.h
  computeShadowFrusta.cpp
  computeShadowFrusta.h
  shadowFrustaShader.cpp
  shadowFrustaShader.h
  rasterize.cpp
  rasterize.h
  rasterizeShader.cpp
  rasterizeShader.h
  merge.cpp
  merge.h
  mergeShader.cpp
  mergeShader.h
  extractSilhouettesShader.cpp
  extractSilhouettesShader.h
  extractSilhouettes.cpp
  extractSilhouettes.h
  traverseSilhouettes.cpp
  traverseSilhouettes.h
  traverseSilhouettesShader.cpp
  traverseSilhouettesShader.h
  collisionShader.cpp
  collisionShader.h
  getConfigShader.cpp
  getConfigShader.h
  getEdgePlanesShader.cpp
  getEdgePlanesShader.h
  )

add_subdirectory(debug)

target_include_directories(${PROJECT_NAME} PUBLIC ..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL RSSV_Debug)
