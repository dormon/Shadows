cmake_minimum_required(VERSION 3.12.1)

add_library(CubeShadowMapping STATIC
  CubeShadowMapping.cpp
  Params.cpp
  CubeShadowMapping.h
  ShadowMapToShadowMaskShaders.h
  CreateShadowMapShaders.h
  Params.h
  )
target_include_directories(CubeShadowMapping PUBLIC ..)
target_link_libraries(CubeShadowMapping PUBLIC geGL::geGL)

