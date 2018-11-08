cmake_minimum_required(VERSION 3.12.1)

project(Shadows)

SET(CMAKE_CXX_STANDARD 14)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(REVISION_VERSION 0)

#include(GenerateExportHeader)

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ".")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")


find_package(ArgumentViewer 1.0 CONFIG REQUIRED)
find_package(BasicCamera 1.0 CONFIG REQUIRED)
find_package(Simple3DApp 1.0.0 CONFIG REQUIRED)
find_package(Vars CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

find_package(FreeImage REQUIRED)
find_package(FreeImagePlus REQUIRED)

set(SOURCES
  src/main.cpp
  src/Model.cpp
  src/CameraPath.cpp
  src/CSV.cpp
  src/Shading.cpp
  src/ShadowMethod.cpp
  src/FastAdjacency.cpp
  src/TimeStamp.cpp
  src/SilhouetteShaders.cpp
  src/CSSV/CSSV.cpp
  src/CSSV/DrawSidesShaders.cpp
  src/CSSV/DrawCapsShaders.cpp
  src/CSSV/ExtractSilhouettes.cpp
  src/CSSV/BasicExtractSilhouettes.cpp
  src/CSSV/PlanesExtractSilhouettes.cpp
  src/CSSV/InterleavedPlanesExtractSilhouettes.cpp
  src/CSSV/Param.cpp
  src/CSSV/DrawSidesProgram.cpp
  src/CSSV/DrawCapsProgram.cpp
  src/CSSV/DrawCaps.cpp
  src/CSSV/DrawSides.cpp
  src/CSSV/DrawSidesShaders.cpp
  src/CSSV/DrawCapsShaders.cpp
  src/CubeShadowMapping/CubeShadowMapping.cpp
  src/CubeShadowMapping/Params.cpp
  src/VSSV/VSSV.cpp
  src/VSSV/Params.cpp
  src/VSSV/DrawCaps.cpp
  src/VSSV/DrawSides.cpp
  src/VSSV/DrawSidesUsingPoints.cpp
  src/VSSV/DrawSidesUsingPlanes.cpp
  src/VSSV/DrawSidesUsingAllPlanes.cpp
  src/CSSVSOE.cpp
  src/Sintorn/Sintorn.cpp
  src/Sintorn/Tiles.cpp
  src/Sintorn/ShadowFrustaShaders.cpp
  src/Sintorn/Param.cpp
  src/Sintorn/ShadowFrusta.cpp
  src/Sintorn/HierarchyShaders.cpp
  src/Sintorn/HierarchicalDepth.cpp
  src/RSSV/RSSV.cpp
  src/RSSV/Tiles.cpp
  src/RSSV/BuildStupidHierarchy.cpp
  src/RSSV/ExtractSilhouettes.cpp
  src/RSSV/Params.cpp
  src/RSSV/StupidHierarchyShaders.cpp
  src/RayTracing/nanort.cc
  src/RayTracing/RayTracing.cpp
  src/DrawPrimitive.cpp
  src/ShadowVolumes.cpp
  src/util.cpp
  src/mainUtil.cpp
  src/CameraParam.cpp
  src/TestParam.cpp
  src/SilhouetteShaders.cpp
  src/Function.cpp
  src/FunctionImpl.cpp
  src/createGBuffer.cpp
  src/imguiVars.cpp
  src/Barrier.cpp
  )

set(INCLUDES
  src/CameraPath.h
  src/CSV.h
  src/Deferred.h
  src/Model.h
  src/Shading.h
  src/ShadowMethod.h
  src/FastAdjacency.h
  src/TimeStamp.h
  src/CSSV/Fwd.h
  src/CSSV/CSSV.h
  src/CSSV/DrawSidesShaders.h
  src/CSSV/DrawCapsShaders.h
  src/CSSV/ExtractSilhouetteShader.h
  src/CSSV/ExtractSilhouettes.h
  src/CSSV/BasicExtractSilhouettes.h
  src/CSSV/PlanesExtractSilhouettes.h
  src/CSSV/InterleavedPlanesExtractSilhouettes.h
  src/CSSV/InterleavedPlanesShader.h
  src/CSSV/DrawSidesProgram.h
  src/CSSV/DrawCapsProgram.h
  src/CSSV/DrawCaps.h
  src/CSSV/DrawSides.h
  src/CSSV/Param.h
  src/CubeShadowMapping/CubeShadowMapping.h
  src/CubeShadowMapping/ShadowMapToShadowMaskShaders.h
  src/CubeShadowMapping/CreateShadowMapShaders.h
  src/CubeShadowMapping/Params.h
  src/VSSV/VSSV.h
  src/VSSV/Params.h
  src/VSSV/DrawCaps.h
  src/VSSV/CapsShader.h
  src/VSSV/DrawSides.h
  src/VSSV/DrawSidesUsingPoints.h
  src/VSSV/DrawSidesUsingPointsShader.h
  src/VSSV/DrawSidesUsingPlanes.h
  src/VSSV/DrawSidesUsingPlanesShader.h
  src/VSSV/DrawSidesUsingAllPlanes.h
  src/VSSV/DrawSidesUsingAllPlanesShader.h
  src/CSSVSOE.h
  src/Sintorn/Sintorn.h
  src/Sintorn/Tiles.h
  src/Sintorn/Shaders.h
  src/Sintorn/ShadowFrustaShaders.h
  src/Sintorn/Param.h
  src/Sintorn/ShadowFrusta.h
  src/Sintorn/HierarchyShaders.h
  src/Sintorn/HierarchicalDepth.h
  src/RSSV/RSSV.h
  src/RSSV/Tiles.h
  src/RSSV/Shaders.h
  src/RSSV/BuildHierarchy.h
  src/RSSV/BuildStupidHierarchy.h
  src/RSSV/StupidHierarchyShaders.h
  src/RSSV/ExtractSilhouettes.h
  src/RSSV/ExtractSilhouettesShader.h
  src/RSSV/Params.h
  src/RayTracing/nanort.h
  src/RayTracing/RayTracing.h
  src/BallotShader.h
  src/DrawPrimitive.h
  src/ProgramExtension.h
  src/ShadowVolumes.h
  src/ShadowVolumesShaders.h
  src/SilhouetteShaders.h
  src/util.h
  src/mainUtil.h
  src/CameraParam.h
  src/TestParam.h
  src/Function.h
  src/FunctionImpl.h
  src/createGBuffer.h
  src/imguiVars.h
  src/Barrier.h
  src/Simplex.h
  src/GLSLLine.h
  )

add_executable(${PROJECT_NAME} ${SOURCES} ${INCLUDES})

target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${PROJECT_NAME} PUBLIC Simple3DApp::Simple3DApp BasicCamera::BasicCamera ArgumentViewer::ArgumentViewer assimp::assimp FreeImage::FreeImage FreeImagePlus::FreeImagePlus Vars::Vars)

#include(print_target_properties.cmake)
#print_target_properties(SDL2::SDL2)


function(getSharedLibraries out lib cfg)
  #message("getSharedLibraries(${lib} ${cfg})")

  if(TARGET ${lib})
    get_target_property(target_type ${lib} TYPE)
    if(${target_type} STREQUAL "INTERFACE_LIBRARY")
      #message("${lib} JE ${target_type}")
      return()
    endif()
  else()
    #message("${lib} NENI TARGET")
    return()
  endif()


  get_target_property(dll ${lib} IMPORTED_LOCATION_${cfg})
  list(APPEND dlls ${dll})

  get_target_property(interfaceLibs ${lib} INTERFACE_LINK_LIBRARIES)
  if(NOT "${interfaceLibs}" STREQUAL "interfaceLibs-NOTFOUND")
    foreach(interfaceLib ${interfaceLibs})
      getSharedLibraries(ilibs ${interfaceLib} ${cfg})
      list(APPEND dlls ${ilibs})
    endforeach()
  endif()
  list(REMOVE_DUPLICATES dlls)
  set(${out} ${dlls} PARENT_SCOPE)
endfunction()


function(getAllSharedLibraries allLibraries app cfg)
  get_target_property(libs ${app} LINK_LIBRARIES)
  foreach(lib ${libs})
    getSharedLibraries(libList ${lib} ${cfg})
    #message("${lib} ###### ${libList}")
    list(APPEND allLibs ${libList})
  endforeach()
  list(REMOVE_DUPLICATES allLibs)
  set(${allLibraries} ${allLibs} PARENT_SCOPE)
endfunction()

getAllSharedLibraries(allDebugSharedLibraries ${PROJECT_NAME} DEBUG)
getAllSharedLibraries(allReleaseSharedLibraries ${PROJECT_NAME} RELEASE)

#message("SharedDebug  : ${allDebugSharedLibraries}")
#message("SharedRelease: ${allReleaseSharedLibraries}")

if(MSVC)
  foreach(lib ${allDebugSharedLibraries})
    file(COPY ${lib} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
  endforeach()
  foreach(lib ${allReleaseSharedLibraries})
    file(COPY ${lib} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
  endforeach()
endif() 

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
install(FILES ${allDebugSharedLibraries} DESTINATION .)
