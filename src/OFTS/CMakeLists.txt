cmake_minimum_required(VERSION 3.12.1)

add_library(OFTS STATIC
	OFTS.h
	OFTS.cpp
	OFTS_params.h
	OFTS_params.cpp
	OFTS_shaderGen.h
	OFTS_shaderGen.cpp
	Frustum.h
	Frustum.cpp
	Plane.h
	Plane.cpp	
)
  
target_include_directories(OFTS PUBLIC ..)
target_link_libraries(OFTS PUBLIC geGL::geGL)
