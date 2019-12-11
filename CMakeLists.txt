cmake_minimum_required(VERSION 3.12.1)

project(Shadows)

SET(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1 ) 

set(MAJOR_VERSION 1)
set(MINOR_VERSION 0)
set(REVISION_VERSION 0)

#include(GenerateExportHeader)

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ".")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
SET(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
SET(CMAKE_RELWITHDEBINFO_POSTFIX "rd" CACHE STRING "add a postfix, usually empty on windows")
SET(CMAKE_MINSIZEREL_POSTFIX "s" CACHE STRING "add a postfix, usually empty on windows")

find_package(ArgumentViewer 1.0 CONFIG REQUIRED)
find_package(BasicCamera 1.0 CONFIG REQUIRED)
find_package(Simple3DApp 1.0.0 CONFIG REQUIRED)
find_package(Vars CONFIG REQUIRED)
find_package(imguiVars CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

find_package(FreeImage REQUIRED)
find_package(FreeImagePlus REQUIRED)

SET(OUTPUT_LIBDIR ${PROJECT_BINARY_DIR}/lib)
SET(OUTPUT_BINDIR ${PROJECT_BINARY_DIR}/bin)

FOREACH(CONF ${CMAKE_CONFIGURATION_TYPES})        # For each configuration (Debug, Release, MinSizeRel... and/or anything the user chooses)
   STRING(TOUPPER "${CONF}" CONF)                # Go uppercase (DEBUG, RELEASE...)
   SET("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_LIBDIR}")
   SET("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_BINDIR}")
   IF(WIN32)
       SET("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_LIBDIR}")
   ELSE()
       SET("CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF}" "${OUTPUT_LIBDIR}")
   ENDIF()
ENDFOREACH()

add_subdirectory(src/Sintorn)
add_subdirectory(src/CSSV)
add_subdirectory(src/VSSV)
add_subdirectory(src/RSSV)
add_subdirectory(src/CubeShadowMapping)
add_subdirectory(src/GSSV)
add_subdirectory(src/TSSV)
add_subdirectory(src/SM)
add_subdirectory(src/DPM)
add_subdirectory(src/ODPM)
add_subdirectory(src/DPSV)
add_subdirectory(src/HSSV)
add_subdirectory(src/Sintorn2)

add_library(RayTracing STATIC
  src/RayTracing/nanort.cc
  src/RayTracing/RayTracing.cpp
  src/RayTracing/nanort.h
  src/RayTracing/RayTracing.h
  )
target_include_directories(RayTracing PUBLIC src)
target_link_libraries(RayTracing PUBLIC glm geGL::geGL)


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
  src/CSSVSOE.cpp
  src/BallotShader.cpp
  src/DrawPrimitive.cpp
  src/ShadowVolumes.cpp
  src/util.cpp
  src/mainUtil.cpp
  src/CameraParam.cpp
  src/loadTestParams.cpp
  src/SilhouetteShaders.cpp
  src/Function.cpp
  src/FunctionImpl.cpp
  src/renderModelToGBuffer.cpp
  src/Barrier.cpp
  src/Methods.cpp
  src/modelStats.cpp
  src/selectMethod.cpp
  src/createShadowMask.cpp
  src/createMethod.cpp
  src/stopAppAfterMaxFrame.cpp
  src/takeAScreenShot.cpp
  src/initMethods.cpp
  src/parseArguments.cpp
  src/loadBasicApplicationParameters.cpp
  src/getMethodNameList.cpp
  src/loadCameraParams.cpp
  src/measureFly.cpp
  src/drawScene.cpp
  src/ifExistBeginStamp.cpp
  src/ifExistStamp.cpp
  src/ifExistEndStamp.cpp
  src/doShading.cpp
  src/ifMethodExistCreateShadowMask.cpp
  src/ifMethodExistsDrawDebug.cpp
  src/saveTexture.cpp
  src/createGeometryBuffer.cpp
  src/copyTexture.cpp
  src/saveGBufferAsPointCloud.cpp
  src/drawPointCloud.cpp
  src/getMVP.cpp
  src/getCameraNear.cpp
  src/getCameraFar.cpp
  src/divRoundUp.cpp
  src/requiredBits.cpp
  src/startStop.cpp
  src/split.cpp
  src/GSCaps.cpp
  src/createAdjacency.cpp
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
  src/CSSVSOE.h
  src/BallotShader.h
  src/DrawPrimitive.h
  src/ProgramExtension.h
  src/ShadowVolumes.h
  src/ShadowVolumesShaders.h
  src/SilhouetteShaders.h
  src/util.h
  src/mainUtil.h
  src/CameraParam.h
  src/loadTestParams.h
  src/Function.h
  src/FunctionImpl.h
  src/renderModelToGBuffer.h
  src/Barrier.h
  src/Simplex.h
  src/GLSLLine.h
  src/Methods.h
  src/FunctionPrologue.h
  src/AABB.h
  src/modelStats.h
  src/selectMethod.h
  src/createShadowMask.h
  src/createMethod.h
  src/stopAppAfterMaxFrame.h
  src/takeAScreenShot.h
  src/initMethods.h
  src/parseArguments.h
  src/loadBasicApplicationParameters.h
  src/getMethodNameList.h
  src/loadCameraParams.h
  src/measureFly.h
  src/drawScene.h
  src/ifExistBeginStamp.h
  src/ifExistStamp.h
  src/ifExistEndStamp.h
  src/doShading.h
  src/ifMethodExistCreateShadowMask.h
  src/ifMethodExistsDrawDebug.h
  src/saveTexture.h
  src/createGeometryBuffer.h
  src/copyTexture.h
  src/saveGBufferAsPointCloud.h
  src/drawPointCloud.h
  src/getMVP.h
  src/getCameraNear.h
  src/getCameraFar.h
  src/divRoundUp.h
  src/requiredBits.h
  src/startStop.h
  src/toStr.h
  src/split.h
  src/fillValues.h
  src/GSCaps.h
  src/GSCapsShaders.h
  src/createAdjacency.h
  src/Timer.h
  src/align.h
  )

add_executable(${PROJECT_NAME} ${SOURCES} ${INCLUDES})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(${PROJECT_NAME} PUBLIC Simple3DApp::Simple3DApp BasicCamera::BasicCamera ArgumentViewer::ArgumentViewer assimp::assimp FreeImage::FreeImage FreeImagePlus::FreeImagePlus Vars::Vars imguiVars::imguiVars)
target_link_libraries(${PROJECT_NAME} PUBLIC Sintorn CSSV VSSV RSSV GSSV TSSV SM DPM ODPM DPSV HSSV CubeShadowMapping RayTracing Sintorn2)

#include(print_target_properties.cmake)
#print_target_properties(SDL2::SDL2)

#add_custom_target(sintorn2 ./Shadows --model /media/devel/models/sponza/sponza.obj --method sintorn2 --window-size 999 777)

function(getSharedLibraries out lib cfg)
  #message("getSharedLibraries(${lib} ${cfg})")

  if(TARGET ${lib})
    get_target_property(target_type ${lib} TYPE)
    #message("TARGET ${lib} ${target_type}")
    if(${target_type} STREQUAL "INTERFACE_LIBRARY")
      #get_target_property(jojo ${lib} INTERFACE_LINK_LIBRARIES)
      #message("${lib} ma ${jojo}")
      #message("${lib} JE ${target_type}")
      return()
    endif()
    if(${target_type} STREQUAL "STATIC_LIBRARY")
      #message("${lib} je ${target_type}")
      return()
    endif()
  else()
    #message("${lib} NENI TARGET")
    return()
  endif()


  get_target_property(dll ${lib} IMPORTED_LOCATION_${cfg})
  #message(">>> ${dll}")
  list(APPEND dlls ${dll})

  get_target_property(interfaceLibs ${lib} INTERFACE_LINK_LIBRARIES)
  #message("--- ${interfaceLibs}")
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
	message("Copying " ${lib})
    file(COPY ${lib} DESTINATION ${OUTPUT_BINDIR})
  endforeach()
  foreach(lib ${allReleaseSharedLibraries})
	message("Copying " ${lib})
    file(COPY ${lib} DESTINATION ${OUTPUT_BINDIR})
  endforeach()
endif() 

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
install(FILES ${allDebugSharedLibraries} DESTINATION .)
