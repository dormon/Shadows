cmake_minimum_required(VERSION 3.12.1)

add_library(DPM STATIC
	DeepPrimitiveMap.cpp
	DeepPrimitiveMap.h
	DpmParams.h
	DpmParams.cpp
	DpmShaderGen.h
	DpmShaderGen.cpp
)
  
target_include_directories(DPM PUBLIC ..)
target_link_libraries(DPM PUBLIC geGL::geGL)
