#pragma once

#include <AABB.h>
#include <Defines.h>

#include <unordered_map>
#include <bitset>

/*
typedef unsigned char BitmaskType;
//typedef uint64_t BitmaskType;

constexpr size_t BitmaskTypeSizeBits = 8 * sizeof(BitmaskType);
constexpr BitmaskType BitmaskAllSet = BitmaskType(-1);
*/

//typedef std::unordered_map<BitSet, std::vector<uint32_t>, BitSetHashFunction> NodeMap;
typedef std::unordered_map<u8, std::vector<u32>> NodeMap;
struct Node
{
	AABB volume;

	NodeMap edgesAlwaysCastMap;
	NodeMap edgesMayCastMap;

	bool isValid() const;
	void clear();

	void shrinkEdgeVectors();
	void sortEdgeVectors();
};