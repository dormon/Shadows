cmake_minimum_required(VERSION 3.12.1)

add_library(CSSV STATIC
  CSSV.cpp
  DrawSidesShaders.cpp
  DrawCapsShaders.cpp
  ExtractSilhouettes.cpp
  BasicExtractSilhouettes.cpp
  PlanesExtractSilhouettes.cpp
  InterleavedPlanesExtractSilhouettes.cpp
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
  ExtractSilhouettes.h
  BasicExtractSilhouettes.h
  PlanesExtractSilhouettes.h
  InterleavedPlanesExtractSilhouettes.h
  InterleavedPlanesShader.h
  DrawSidesProgram.h
  DrawCapsProgram.h
  DrawCaps.h
  DrawSides.h
  Param.h

  )
target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL)
