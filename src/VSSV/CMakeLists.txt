cmake_minimum_required(VERSION 3.12.1)

add_library(VSSV STATIC
  VSSV.cpp
  Params.cpp
  DrawCaps.cpp
  DrawSides.cpp
  DrawSidesUsingPoints.cpp
  DrawSidesUsingPlanes.cpp
  DrawSidesUsingAllPlanes.cpp
  VSSV.h
  Params.h
  DrawCaps.h
  CapsShader.h
  DrawSides.h
  DrawSidesUsingPoints.h
  DrawSidesUsingPointsShader.h
  DrawSidesUsingPlanes.h
  DrawSidesUsingPlanesShader.h
  DrawSidesUsingAllPlanes.h
  DrawSidesUsingAllPlanesShader.h
  )
target_include_directories(VSSV PUBLIC ..)
target_link_libraries(VSSV PUBLIC geGL::geGL)

