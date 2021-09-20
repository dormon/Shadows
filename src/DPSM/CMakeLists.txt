cmake_minimum_required(VERSION 3.12.1)

add_library(DPSM STATIC
	DPSM.h
	DPSM.cpp
	DPSM_params.h
	DPSM_params.cpp
	DPSM_shaders.h
	DPSM_shaders.cpp
)
  
target_include_directories(DPSM PUBLIC ..)
target_link_libraries(DPSM PUBLIC geGL::geGL)
