cmake_minimum_required(VERSION 3.12.1)

project(Sintorn2_Debug)

add_library(${PROJECT_NAME} STATIC
  drawDebug.cpp
  drawDebug.h
  dumpData.cpp
  dumpData.h
  drawSamples.cpp
  drawSamples.h
  drawNodePool.cpp
  drawNodePool.h
  )

target_include_directories(${PROJECT_NAME} PUBLIC .. ../..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL)
