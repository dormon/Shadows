#pragma once

#include <Defines.h>

#include <string>
#include <vector>

struct EdgeRangeShaderParams
{
	u32 maxOctreeLevel;
	u32 nofSubbuffers;
	u32 subbufferCorrection;
	u32 bitmaskBufferSize;
};

struct SidesGenShaderParams
{
	u32 nofBitsMultiplicity;
	u32 bitmaskBufferSize;
	u32 wgSize;
	u32 maxOctreeLevel;
};

struct SidesGenShaderParams2
{
	u32 nofBitsMultiplicity;
	u32 maxMultiplicity;
	u32 bitmaskBufferSize;
	u32 wgSize;
	u32 edgeSizeNofVec4;
};

std::string getComputeEdgeRangesCsSource(std::vector<u32> const& lastNodePerBuffer, EdgeRangeShaderParams const& params);

std::string getComputeSidesFromEdgeRangesCsSource(std::vector<u32> const& lastNodePerBuffer, SidesGenShaderParams const& params);
std::string getComputeSidesFromEdgeRangesCsSource2(std::vector<u32> const& lastNodePerBuffer, SidesGenShaderParams2 const& params);

std::string genEdgeBuffersMappingFn(std::vector<u32> const& lastNodePerBuffer);