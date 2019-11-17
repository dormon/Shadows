cmake_minimum_required(VERSION 3.12.1)

add_library(CSSV STATIC
  CSSV.cpp
  DrawSidesShaders.cpp
  DrawCapsShaders.cpp
  Param.cpp
  DrawSidesProgram.cpp
  DrawCapsProgram.cpp
  DrawCaps.cpp
  DrawSides.cpp
  DrawSidesShaders.cpp
  DrawCapsShaders.cpp
  Fwd.h
  CSSV.h
  DrawSidesShaders.h
  DrawCapsShaders.h
  ExtractSilhouetteShader.h
  InterleavedPlanesShader.h
  DrawSidesProgram.h
  DrawCapsProgram.h
  DrawCaps.h
  DrawSides.h
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

  )
target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL)
