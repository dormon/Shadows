#pragma once

#include <AABB.h>

#include <unordered_map>


typedef unsigned char BitmaskType;
//typedef uint64_t BitmaskType;

constexpr size_t BitmaskTypeSizeBits = 8 * sizeof(BitmaskType);
constexpr BitmaskType BitmaskAllSet = BitmaskType(-1);

struct Node
{
	AABB volume;

	std::unordered_map<BitmaskType, std::vector<uint32_t>> edgesAlwaysCastMap;
	std::unordered_map<BitmaskType, std::vector<uint32_t>> edgesMayCastMap;

	bool isValid() const;
	void clear();

	void shrinkEdgeVectors();
	void sortEdgeVectors();
};