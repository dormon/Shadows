cmake_minimum_required(VERSION 3.12.1)

add_library(Sintorn STATIC 
  Sintorn.cpp
  Tiles.cpp
  ShadowFrustaShaders.cpp
  Param.cpp
  ComputeTileSizes.cpp
  ShadowFrusta.cpp
  HierarchyShaders.cpp
  HierarchicalDepth.cpp
  MergeShaders.cpp
  MergeStencil.cpp
  RasterizationShaders.cpp
  Rasterize.cpp
  Sintorn.h
  Tiles.h
  Shaders.h
  ShadowFrustaShaders.h
  Param.h
  ComputeTileSizes.h
  ShadowFrusta.h
  HierarchyShaders.h
  HierarchicalDepth.h
  MergeShaders.h
  MergeStencil.h
  RasterizationShaders.h
  Rasterize.h
  )
target_include_directories(Sintorn PUBLIC ..)
target_link_libraries(Sintorn PUBLIC geGL::geGL)

