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
	Defines.h
	HSSV.cpp
	HSSV.h
	HssvParams.h
	HssvParams.cpp
	HighResolutionTimer.h
	HighResolutionTimer.cpp
	MathOps.h
	MathOps.cpp
	MultiplicityCoder.h
	MultiplicityCoder.cpp
	Node.h
	Node.cpp
	Octree.h
	Octree.cpp
	OctreeSerializer.h
	OctreeSerializer.cpp
	Plane.h
	Plane.cpp
	SidesDrawerBase.h
	SidesDrawerBase.cpp
	CPU/CpuBuilder.h
	CPU/CpuBuilder.cpp
	CPU/CpuPotEdgePropagator.h
	CPU/CpuPotEdgePropagator.cpp
	CPU/CpuSidesDrawer.h
	CPU/CpuSidesDrawer.cpp
	GPU/GpuSidesDrawer.h
	GPU/GpuSidesDrawer.cpp
	GPU/GpuShaderGenerator.h
	GPU/GpuShaderGenerator.cpp
)

source_group("CPU" FILES 
	CPU/CpuBuilder.h 
	CPU/CpuBuilder.cpp
	CPU/CpuPotEdgePropagator.h
	CPU/CpuPotEdgePropagator.cpp
	CPU/CpuSidesDrawer.h
	CPU/CpuSidesDrawer.cpp
)

source_group("GPU" FILES 
	GPU/GpuSidesDrawer.h
	GPU/GpuSidesDrawer.cpp
	GPU/GpuShaderGenerator.h
	GPU/GpuShaderGenerator.cpp
)


target_include_directories(HSSV PUBLIC ..)
target_link_libraries(HSSV PUBLIC geGL::geGL)
