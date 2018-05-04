cmake_minimum_required(VERSION 3.10.1)

project(Shadows)

SET(CMAKE_CXX_STANDARD 14)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(REVISION_VERSION 0)

#include(GenerateExportHeader)

set(APP_NAME Shadows)

find_package(ArgumentViewer 1.0 CONFIG REQUIRED)
find_package(geGL 1.0 CONFIG REQUIRED)
find_package(SDL2CPP 1.0 CONFIG REQUIRED)
find_package(BasicCamera 1.0 CONFIG REQUIRED)
#find_package(SDL2 CONFIG REQUIRED)

set(SOURCES
  src/main.cpp
  src/Model.cpp
  src/CameraPath.cpp
  src/CSV.cpp
  src/Shading.cpp
  src/ShadowMethod.cpp
  src/CubeShadowMapping.cpp
  src/FastAdjacency.cpp
  src/TimeStamp.cpp
  src/CSSV.cpp
  src/CSSVSOE.cpp
  src/Sintorn.cpp
  src/SintornTiles.cpp
  src/RSSV.cpp
  src/RSSVTiles.cpp
  src/VSSV.cpp
  src/DrawPrimitive.cpp
  src/ShadowVolumes.cpp
  )

set(INCLUDES
  src/CameraPath.h
  src/CSV.h
  src/Deferred.h
  src/Model.h
  src/Shading.h
  src/ShadowMethod.h
  src/CubeShadowMapping.h
  src/CubeShadowMappingShaders.h
  src/FastAdjacency.h
  src/TimeStamp.h
  src/CSSV.h
  src/CSSVShaders.h
  src/CSSVSOE.h
  src/Sintorn.h
  src/SintornTiles.h
  src/SintornShaders.h
  src/RSSV.h
  src/RSSVTiles.h
  src/RSSVShaders.h
  src/VSSV.h
  src/VSSVShaders.h
  src/DrawPrimitive.h
  src/ProgramExtension.h
  src/ShadowVolumes.h
  src/ShadowVolumesShaders.h
  src/SilhouetteShaders.h
  src/util.h
  )

add_executable(${APP_NAME} ${SOURCES} ${INCLUDES})

target_include_directories(${APP_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${APP_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${APP_NAME} PUBLIC SDL2CPP::SDL2CPP BasicCamera::BasicCamera geGL::geGL ArgumentViewer::ArgumentViewer assimp)

#set(LIB_NAME_LOWER)
#string(TOLOWER ${LIB_NAME} LIB_NAME_LOWER)
#
#generate_export_header(${LIB_NAME} EXPORT_FILE_NAME ${LIB_NAME}/${LIB_NAME_LOWER}_export.h)
#
#set(${LIB_NAME}_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${REVISION_VERSION})
#
#set_property(TARGET ${LIB_NAME} PROPERTY VERSION ${${LIB_NAME}_VERSION})
#set_property(TARGET ${LIB_NAME} PROPERTY SOVERSION 1)
#set_property(TARGET ${LIB_NAME} PROPERTY INTERFACE_${LIB_NAME}_MAJOR_VERSION 1)
#set_property(TARGET ${LIB_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${LIB_NAME}_MAJOR_VERSION)
#
#install(TARGETS ${LIB_NAME} EXPORT ${LIB_NAME}Targets
#  LIBRARY DESTINATION lib
#  ARCHIVE DESTINATION lib
#  RUNTIME DESTINATION bin
#  INCLUDES DESTINATION include
#  )
#
#install(
#  FILES
#  ${INCLUDES}
#  ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME_LOWER}_export.h
#  DESTINATION
#  include/${LIB_NAME}
#  COMPONENT
#  Devel
#  )
#
#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#  ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME}ConfigVersion.cmake
#  VERSION ${${LIB_NAME}_VERSION}
#  COMPATIBILITY SameMajorVersion
#  )
#
#export(EXPORT ${LIB_NAME}Targets
#  FILE ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME}Targets.cmake
#  NAMESPACE ${LIB_NAME}::
#  )
#
#set(ConfigPackageLocation lib/cmake/${LIB_NAME})
#install(EXPORT ${LIB_NAME}Targets
#  FILE
#  ${LIB_NAME}Targets.cmake
#  NAMESPACE
#  ${LIB_NAME}::
#  DESTINATION
#  ${ConfigPackageLocation}
#  )
#
#file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME}Config.cmake
#  "include(CMakeFindDependencyMacro)\n"
#  "find_dependency(SDL2)\n"
#  "include($" "{CMAKE_CURRENT_LIST_DIR}/${LIB_NAME}Targets.cmake)\n"
#  )
#
#install(
#  FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME}Config.cmake
#  ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/${LIB_NAME}ConfigVersion.cmake
#  DESTINATION
#  ${ConfigPackageLocation}
#  COMPONENT
#  Devel
#  )

