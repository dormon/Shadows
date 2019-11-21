cmake_minimum_required(VERSION 3.12.1)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(HSSV STATIC
	AdjacencyWrapper.h
	AdjacencyWrapper.cpp
	HSSV.cpp
	HSSV.h
	HssvParams.h
	HssvParams.cpp
	MathOps.h
	MathOps.cpp
	Node.h
	Node.cpp
	Octree.h
	Octree.cpp
	Plane.h
	Plane.cpp
	CPU/CpuBuilder.h
	CPU/CpuBuilder.cpp
)

source_group("CPU" FILES CPU/CpuBuilder.h CPU/CpuBuilder.cpp)

target_include_directories(HSSV PUBLIC ..)
target_link_libraries(HSSV PUBLIC geGL::geGL)
