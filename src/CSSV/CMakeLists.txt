cmake_minimum_required(VERSION 3.12.1)

add_library(CSSV STATIC
  Fwd.h
  ExtractSilhouetteShader.h
  InterleavedPlanesShader.h
  CSSV.cpp
  CSSV.h
  sidesShaders.cpp
  sidesShaders.h
  Param.cpp
  Param.h
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
  createDrawSidesProgram.cpp
  createDrawSidesProgram.h
  createSidesVAO.cpp
  createSidesVAO.h
  drawSides.cpp
  drawSides.h

  caps/shaders.cpp
  caps/shaders.h
  caps/createProgram.cpp
  caps/createProgram.h
  caps/createBuffer.cpp
  caps/createBuffer.h
  caps/createVAO.cpp
  caps/createVAO.h
  caps/draw.cpp
  caps/draw.h


  )
target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL)
