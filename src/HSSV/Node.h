#pragma once

#include <AABB.h>
//#include <BitSet.h>

#include <unordered_map>
#include <bitset>

/*
typedef unsigned char BitmaskType;
//typedef uint64_t BitmaskType;

constexpr size_t BitmaskTypeSizeBits = 8 * sizeof(BitmaskType);
constexpr BitmaskType BitmaskAllSet = BitmaskType(-1);
*/

//typedef std::unordered_map<BitSet, std::vector<uint32_t>, BitSetHashFunction> NodeMap;
typedef std::unordered_map<std::bitset<8>, std::vector<uint32_t>> NodeMap;
struct Node
{
	AABB volume;

	//std::unordered_map<BitmaskType, std::vector<uint32_t>> edgesAlwaysCastMap;
	//std::unordered_map<BitmaskType, std::vector<uint32_t>> edgesMayCastMap;

	NodeMap edgesAlwaysCastMap;
	NodeMap edgesMayCastMap;

	bool isValid() const;
	void clear();

	void shrinkEdgeVectors();
	void sortEdgeVectors();
};