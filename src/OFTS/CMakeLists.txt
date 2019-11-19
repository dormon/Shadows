cmake_minimum_required(VERSION 3.12.1)

add_library(OFTS STATIC
	OmnidirFrustumTracedShadows.h
	OmnidirFrustumTracedShadows.cpp
	OftsParams.h
	OftsParams.cpp
)
  
target_include_directories(OFTS PUBLIC ..)
target_link_libraries(OFTS PUBLIC geGL::geGL)
