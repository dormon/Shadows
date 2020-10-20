cmake_minimum_required(VERSION 3.12.1)

add_library(OFTS STATIC
	OFTS.h
	OFTS.cpp
	OFTS_params.h
	OFTS_params.cpp
)
  
target_include_directories(OFTS PUBLIC ..)
target_link_libraries(OFTS PUBLIC geGL::geGL)
