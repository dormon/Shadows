cmake_minimum_required(VERSION 3.12.1)

add_library(ODPM STATIC
	OmnidirDPM.h
	OmnidirDPM.cpp
	OdpmParams.h
	OdpmParams.cpp
)
  
target_include_directories(ODPM PUBLIC ..)
target_link_libraries(ODPM PUBLIC geGL::geGL)
