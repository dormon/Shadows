#include <CPU/CpuBuilder.h>
#include <Plane.h>
#include <Octree.h>
#include <AdjacencyWrapper.h>

#include <FastAdjacency.h>
#include <Octree.h>
#include <MathOps.h>

#include <omp.h>
#include <stack>
#include <cassert>

constexpr uint8_t ALL_SET = uint8_t(-1);

void CpuBuilder::fillOctree(Octree* octree, Adjacency const* adjacency)
{
	assert(Octree != nullptr);
	this->octree = octree;

	std::vector< std::vector<Plane>> edgePlanes = createEdgePlanes(adjacency);

	size_t const nofEdges = edgePlanes.size();
	
	struct CurrentStatus
	{
		uint32_t currentNode;
		uint8_t currentLevel;
	};

	for(size_t i = 0; i< nofEdges; ++i)
	{
		std::stack<CurrentStatus> nodeStack;
		nodeStack.push({0, 0});

		while(!nodeStack.empty())
		{
			auto const currentStatus = nodeStack.top();
			nodeStack.pop();

			ChildResults results = testChildNodes(octree->getChildrenStartingId(currentStatus.currentNode), i, edgePlanes[i], adjacency);


		}
	}
}

std::vector< std::vector<Plane>> CpuBuilder::createEdgePlanes(Adjacency const* adjacency)
{
	std::vector< std::vector<Plane>> planes;
	size_t const numEdges = adjacency->getNofEdges();

	planes.resize(numEdges);

	uint32_t index = 0;

	for (size_t i = 0; i < numEdges; ++i)
	{
		const auto nofOpposites = getNofOppositeVertices(adjacency, i);
		planes[index].reserve(nofOpposites);

		const glm::vec3& v1 = getEdgeVertexLow(adjacency, i);
		const glm::vec3& v2 = getEdgeVertexHigh(adjacency, i);

		for (uint32_t j = 0; j < nofOpposites; ++j)
		{
			Plane p;
			p.createFromPointsCCW(v1, getOppositeVertex(adjacency, i, j), v2);

			planes[index].push_back(p);
		}

		++index;
	}

	return planes;
}

CpuBuilder::ChildResults CpuBuilder::testChildNodes(uint32_t firstChild, uint32_t edgeID, std::vector<Plane> const& edgePlanes, Adjacency const* adjacency)
{
	ChildResults results;
	uint32_t const nofOpposite = getNofOppositeVertices(adjacency, edgeID);

	for(uint32_t child = 0; child < OCTREE_NUM_CHILDREN; ++child)
	{		
		uint32_t const currentNode = firstChild + child;

		const bool isPotentiallySilhouette = (nofOpposite > 1) && MathOps::isEdgeSpaceAaabbIntersecting(edgePlanes, octree->getNodeVolume(currentNode));
		
		if(isPotentiallySilhouette)
		{
			results.potMask[child] = true;
		}
		else
		{
			int const multiplicity = MathOps::calcEdgeMultiplicity(adjacency, edgeID, octree->getNodeVolume(currentNode).getMin());

			if (multiplicity != 0)
			{
				results.silMasks[multiplicity][child] = true;
			}
		}
	}

	return results;
}

void CpuBuilder::assignSilhouetteEdges(std::unordered_map<int, std::bitset<8>> const& results, uint32_t edgeId, uint32_t nodeId)
{

}
