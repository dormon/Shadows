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

  caps/capsShaders.cpp
  caps/capsShaders.h
  caps/createCapsProgram.cpp
  caps/createCapsProgram.h
  caps/createCapsBuffer.cpp
  caps/createCapsBuffer.h
  caps/createCapsVAO.cpp
  caps/createCapsVAO.h
  caps/drawCaps.cpp
  caps/drawCaps.h


  )
target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL)
