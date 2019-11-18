cmake_minimum_required(VERSION 3.12.1)

add_library(GSSV STATIC
	GenerateGeometryShaderSilhouette.cpp
	GenerateGeometryShaderSilhouette.hpp	
	GSSV.hpp
	GSSV.cpp
	GSSV_Shaders.h
	GSSV_params.h
	GSSV_params.cpp
)
  
target_include_directories(GSSV PUBLIC ..)
target_link_libraries(GSSV PUBLIC geGL::geGL)
