cmake_minimum_required(VERSION 3.12.1)

add_library(MTSV STATIC
	MTSV.h
	MTSV.cpp
	MTSV_params.h
	MTSV_params.cpp
	MTSV_shaders.h
	MTSV_shaders.cpp
)
  
target_include_directories(MTSV PUBLIC ..)
target_link_libraries(MTSV PUBLIC geGL::geGL)
