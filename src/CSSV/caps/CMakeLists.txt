cmake_minimum_required(VERSION 3.12.1)

project(CSSV_caps)

add_library(${PROJECT_NAME} STATIC
  shaders.cpp
  shaders.h
  createProgram.cpp
  createProgram.h
  createBuffer.cpp
  createBuffer.h
  createVAO.cpp
  createVAO.h
  draw.cpp
  draw.h
  )

target_include_directories(${PROJECT_NAME} PUBLIC .. ../..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL)
