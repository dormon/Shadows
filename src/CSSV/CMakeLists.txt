cmake_minimum_required(VERSION 3.12.1)

add_library(CSSV STATIC
  Fwd.h
  CSSV.cpp
  CSSV.h
  Param.cpp
  Param.h
  )

add_subdirectory(caps)
add_subdirectory(sides)

target_include_directories(CSSV PUBLIC ..)
target_link_libraries(CSSV PUBLIC geGL::geGL CSSV_caps CSSV_sides)
