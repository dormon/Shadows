cmake_minimum_required(VERSION 3.12.1)

project(Sintorn2)

add_library(${PROJECT_NAME} STATIC
  Sintorn2.cpp
  Sintorn2.h
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
  drawDebug.cpp
  drawDebug.h
  )

target_include_directories(${PROJECT_NAME} PUBLIC ..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL)
