cmake_minimum_required(VERSION 3.12.1)

add_library(CSSV STATIC
  Fwd.h
  ExtractSilhouetteShader.h
  InterleavedPlanesShader.h
  CSSV.cpp
  CSSV.h
  DrawCapsShaders.cpp
  DrawCapsShaders.h
  DrawSidesShaders.cpp
  DrawSidesShaders.h
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
  createDrawSidesVAO.cpp
  createDrawSidesVAO.h
  drawSides.cpp
  drawSides.h
  createCapsProgram.cpp
  createCapsProgram.h
  createCapsBuffer.cpp
  createCapsBuffer.h
  createCapsVAO.cpp
  createCapsVAO.h
  drawCaps.cpp
  drawCaps.h


  )
target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL)
