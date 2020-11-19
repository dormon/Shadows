cmake_minimum_required(VERSION 3.12.1)

add_library(FTS STATIC
	FTS.h
	FTS.cpp
	FTS_params.h
	FTS_params.cpp
	FTS_shaderGen.h
	FTS_shaderGen.cpp
)
  
target_include_directories(FTS PUBLIC ..)
target_link_libraries(FTS PUBLIC geGL::geGL)
