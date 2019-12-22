cmake_minimum_required(VERSION 3.12.1)

SET(CMAKE_CXX_STANDARD 17)

project(Sintorn2_Debug)

add_library(${PROJECT_NAME} OBJECT
  drawDebug.cpp
  drawDebug.h
  dumpData.cpp
  dumpData.h
  drawSamples.cpp
  drawSamples.h
  drawNodePool.cpp
  drawNodePool.h
  drawTraverse.cpp
  drawTraverse.h
  drawSF.cpp
  drawSF.h
  )

target_include_directories(${PROJECT_NAME} PUBLIC .. ../..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL)
