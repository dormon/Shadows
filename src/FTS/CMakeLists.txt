cmake_minimum_required(VERSION 3.12.1)

add_library(FTS STATIC
	FrustumTracedShadows.cpp
	FrustumTracedShadows.h
	FtsParams.h
	FtsParams.cpp
	FtsShaderGen.h
	FtsShaderGen.cpp
)
  
target_include_directories(FTS PUBLIC ..)
target_link_libraries(FTS PUBLIC geGL::geGL)
