cmake_minimum_required(VERSION 3.12.1)

find_package(OpenMP)

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
	GPU/GpuSidesDrawer2.h
	GPU/GpuSidesDrawer2.cpp
	GPU/GpuSidesDrawer3.h
	GPU/GpuSidesDrawer3.cpp
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
	GPU/GpuSidesDrawer2.h
	GPU/GpuSidesDrawer2.cpp
	GPU/GpuSidesDrawer3.h
	GPU/GpuSidesDrawer3.cpp
)


target_include_directories(HSSV PUBLIC ..)
target_link_libraries(HSSV PUBLIC geGL::geGL OpenMP::OpenMP_CXX)
