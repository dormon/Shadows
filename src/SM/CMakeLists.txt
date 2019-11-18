cmake_minimum_required(VERSION 3.12.1)

add_library(SM STATIC
	ShadowMapping.h
	ShadowMapping.cpp
	ShadowMappingParams.h
	ShadowMappingParams.cpp
	ShadowMappingShaders.h
)
  
target_include_directories(SM PUBLIC ..)
target_link_libraries(SM PUBLIC geGL::geGL)