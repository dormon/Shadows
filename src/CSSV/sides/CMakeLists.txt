cmake_minimum_required(VERSION 3.12.1)

project(CSSV_sides)

add_library(${PROJECT_NAME} STATIC
  extractShader.h
  createExtractProgram.cpp
  createExtractProgram.h
  createInterleavedPlanesEdges.cpp
  createInterleavedPlanesEdges.h
  createPlanesEdges.cpp
  createPlanesEdges.h
  createBasicEdges.cpp
  createBasicEdges.h
  createSilhouetteBuffer.cpp
  createSilhouetteBuffer.h
  createDIBO.cpp
  createDIBO.h
  extractSilhouettes.cpp
  extractSilhouettes.h
  createDrawProgram.cpp
  createDrawProgram.h
  createVAO.cpp
  createVAO.h
  draw.cpp
  draw.h
  drawShaders.cpp
  drawShaders.h
  )

target_include_directories(${PROJECT_NAME} PUBLIC .. ../..)
target_link_libraries(${PROJECT_NAME} PUBLIC geGL::geGL)
