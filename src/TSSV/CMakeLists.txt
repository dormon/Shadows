cmake_minimum_required(VERSION 3.12.1)

cmake_minimum_required(VERSION 3.12.1)

add_library(TSSV STATIC
	SidesShaderGenerator.cpp
	SidesShaderGenerator.hpp	
	TSSV.hpp
	TSSV.cpp
	TSSV_Params.h
	TSSV_Params.cpp
)
  
target_include_directories(TSSV PUBLIC ..)
target_link_libraries(TSSV PUBLIC geGL::geGL)