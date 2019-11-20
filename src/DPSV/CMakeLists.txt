cmake_minimum_required(VERSION 3.12.1)

add_library(DPSV STATIC
	DpsvParams.h
	DpsvParams.cpp
	DPSV.h
	DPSV.cpp
	DpsvShaders.h
	DpsvShaders.cpp
)
  
target_include_directories(DPSV PUBLIC ..)
target_link_libraries(DPSV PUBLIC geGL::geGL)
