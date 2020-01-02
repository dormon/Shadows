cmake_minimum_required(VERSION 3.12.1)

add_library(RSSV STATIC
  RSSV.cpp
  Tiles.cpp
  BuildStupidHierarchy.cpp
  ExtractSilhouettes.cpp
  Params.cpp
  StupidHierarchyShaders.cpp
  PerfectResolution/Build.cpp
  Hierarchy.cpp
  RSSV.h
  Tiles.h
  Shaders.h
  BuildHierarchy.h
  BuildStupidHierarchy.h
  StupidHierarchyShaders.h
  Hierarchy.h
  ExtractSilhouettes.h
  ExtractSilhouettesShader.h
  Params.h
  PerfectResolution/Build.h
  )
target_include_directories(RSSV PUBLIC ..)
target_link_libraries(RSSV PUBLIC geGL::geGL)
