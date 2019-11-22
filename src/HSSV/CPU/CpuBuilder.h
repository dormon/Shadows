#pragma once

#include <Plane.h>

#include <bitset>
#include <vector>
#include <unordered_map>

class Octree;
class Adjacency;

class CpuBuilder
{
public:
	void fillOctree(Octree* octree, Adjacency const* adjacency);

protected:

	Octree* octree;

	struct ChildResults
	{
		std::bitset<8> potMask;
		std::unordered_map<int, std::bitset<8>> silMasks; //for each multiplicity
	};

	std::vector< std::vector<Plane>> createEdgePlanes(Adjacency const* adjacency);
	
	ChildResults testChildNodes(uint32_t firstChild, uint32_t edgeID, std::vector<Plane> const& edgePlanes, Adjacency const* adjacency);

	void assignSilhouetteEdges(std::unordered_map<int, std::bitset<8>> const& results, uint32_t edgeId, uint32_t nodeId);
};