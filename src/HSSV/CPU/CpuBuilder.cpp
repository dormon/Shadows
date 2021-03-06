#include <CPU/CpuBuilder.h>
#include <CPU/CpuPotEdgePropagator.h>

#include <Plane.h>
#include <Octree.h>
#include <AdjacencyWrapper.h>
#include <MultiplicityCoder.h>
#include <HighResolutionTimer.h>

#include <FastAdjacency.h>
#include <Octree.h>
#include <MathOps.h>

#include <omp.h>

#include <cassert>
#include <iostream>

#define ENABLE_MULTITHREAD

void CpuBuilder::fillOctree(Octree* o, Adjacency const* adjacency, u32 multiplicityBits, bool compressed, u32 minimumNofEdgesNode)
{
	assert(o != nullptr);
	assert(multiplicityBits > 1);
	assert(multiplicityBits < 31);

	octree = o;
	NofBitsMultiplicity = multiplicityBits;
	IsCompressed = compressed;

	std::vector< std::vector<Plane>> edgePlanes = createEdgePlanes(adjacency);
	s32 const nofEdges = s32(edgePlanes.size());
	
	std::cerr << "Octree construction started, nof edges: " << nofEdges << std::endl;
	
	HighResolutionTimer t;
	
#ifdef ENABLE_MULTITHREAD
	createAllOctreeMasks();
	Mutexes = std::make_unique<std::mutex[]>(2 * 256 * octree->getTotalNumNodes());
	#pragma omp parallel for
#endif
	for(s32 i = 0; i< nofEdges; ++i)
	{
		processEdge(edgePlanes[i], u32(i), adjacency);
	}

#ifdef ENABLE_MULTITHREAD
	removeEmptyMasks();
#endif
	
	std::cerr << "Edges added in " << float(t.getElapsedTimeFromLastQuerySeconds()) << "s\n";

	propagatePotEdges();

	std::cerr << "Pot edges propagated in " << float(t.getElapsedTimeFromLastQuerySeconds()) << "s\n";
	
	if(!IsCompressed && minimumNofEdgesNode>0)
	{
		discardNodesWithLessThanEdges(256);

		std::cerr << "insufficiently big buffers merged in " << float(t.getElapsedTimeFromLastQuerySeconds()) << "s\n";
	}
	
	octree->makeNodesFit();

	std::cerr << "Octree optimally allocated in " << float(t.getElapsedTimeFromLastQuerySeconds()) << "s\n";
	std::cerr << "Octree build in " << float(t.getElapsedTimeSeconds()) << "s, size " << (octree->getOctreeSizeBytes() >> 20) << "MB\n";
}

std::vector< std::vector<Plane>> CpuBuilder::createEdgePlanes(Adjacency const* adjacency)
{
	std::vector< std::vector<Plane>> planes;
	s32 const numEdges = s32(adjacency->getNofEdges());

	planes.resize(numEdges);

	#pragma omp parallel for
	for (s32 i = 0; i < numEdges; ++i)
	{
		const auto nofOpposites = getNofOppositeVertices(adjacency, i);
		planes[i].reserve(nofOpposites);

		const glm::vec3& v1 = getEdgeVertexLow(adjacency, i);
		const glm::vec3& v2 = getEdgeVertexHigh(adjacency, i);

		for (u32 j = 0; j < nofOpposites; ++j)
		{
			Plane p;
			p.createFromPointsCCW(v1, getOppositeVertex(adjacency, i, j), v2);

			planes[i].push_back(p);
		}
	}

	return planes;
}

void CpuBuilder::processEdge(std::vector<Plane> const& edgePlanes, u32 edgeId, Adjacency const* ad)
{
	Stack statusStack;
	statusStack.push({ 0, 0 });

	while (!statusStack.empty())
	{
		CurrentStatus const status = statusStack.top();
		statusStack.pop();

		ChildResults results = testChildNodes(octree->getChildrenStartingId(status.currentNode), edgeId, edgePlanes, ad);

		processSilhouetteEdge(status, edgeId, results.silMasks);

		processPotentialEdge(status, edgeId, results.potMask, statusStack);
	}
}

CpuBuilder::ChildResults CpuBuilder::testChildNodes(u32 firstChild, u32 edgeID, std::vector<Plane> const& edgePlanes, Adjacency const* adjacency)
{
	ChildResults results;

	for(u32 child = 0; child < OCTREE_NUM_CHILDREN; ++child)
	{		
		u32 const currentNode = firstChild + child;

		const bool isPotentiallySilhouette = MathOps::isEdgeSpaceAaabbIntersecting(edgePlanes, octree->getNodeVolume(currentNode));
		
		if(isPotentiallySilhouette)
		{
			results.potMask[child] = true;
		}
		else
		{
			s32 const multiplicity = MathOps::calcEdgeMultiplicity(adjacency, edgeID, glm::vec4(octree->getNodeVolume(currentNode).getMin(), 1));

			if (multiplicity != 0)
			{
				results.silMasks[multiplicity][child] = true;
			}
		}
	}

	return results;
}

void CpuBuilder::processSilhouetteEdge(CurrentStatus const& status, u32 edgeId, std::unordered_map<s32, BitSet8> const& silMasks)
{
	if(silMasks.size()==0)
	{
		return;
	}
	
	if(IsCompressed)
	{
		storeSilhouetteEdgesCompressed(silMasks, edgeId, status.currentNode);
	}
	else
	{
		storeSilhouetteEdges(silMasks, edgeId, status.currentNode);
	}
}

void CpuBuilder::storeSilhouetteEdgesCompressed(std::unordered_map<s32, BitSet8> const& silMasks, u32 edgeId, u32 nodeId)
{
	MultiplicityCoder coder(NofBitsMultiplicity);

	for(auto const& silMask : silMasks)
	{
		u8 const mask = u8(silMask.second.to_ulong());
		u32 const encodedEdge = coder.encodeEdgeMultiplicityToId(edgeId, silMask.first);

		if(silMask.second.count() > 1)
		{
			storeSilhouetteEdge(encodedEdge, nodeId, mask);
		}
		else
		{
			s8 const pos = MathOps::findFirstSet(mask);
			assert(pos >= 0);

			storeSilhouetteEdge(encodedEdge, octree->getChildrenStartingId(nodeId) + pos, BITMASK_ALL_SET);
		}
	}	
}

void CpuBuilder::storeSilhouetteEdges(std::unordered_map<s32, BitSet8> const& silMasks, u32 edgeId, u32 nodeId)
{
	MultiplicityCoder coder(NofBitsMultiplicity);

	for (auto const& silMask : silMasks)
	{
		u8 const mask = u8(silMask.second.to_ulong());
		u32 const encodedEdge = coder.encodeEdgeMultiplicityToId(edgeId, silMask.first);

		if(mask==BITMASK_ALL_SET)
		{
			//Store in parent node
			storeSilhouetteEdge(encodedEdge, nodeId, BITMASK_ALL_SET);
		}
		else
		{
			//Store in children
			u32 const startChild = octree->getChildrenStartingId(nodeId);

			for(u32 i = 0; i< OCTREE_NUM_CHILDREN; ++i)
			{
				if(silMask.second.test(i))
				{
					storeSilhouetteEdge(encodedEdge, startChild + i, BITMASK_ALL_SET);
				}
			}
		}
	}
}

void CpuBuilder::storePotentialEdgesCompressed(BitSet8 const& potMask, u32 edgeId, u32 parentNode)
{
	size_t const count = potMask.count();
	u8 const bitMask = u8(potMask.to_ulong());

	if (count > 1)
	{
		//store in current node
		storePotentialEdge(edgeId, parentNode, bitMask);
	}
	else if (count == 1)
	{
		//store in the child
		s8 pos = MathOps::findFirstSet(bitMask);
		assert(pos >= 0);

		storePotentialEdge(edgeId, octree->getChildrenStartingId(parentNode) + u8(pos), BITMASK_ALL_SET);
	}
}

void CpuBuilder::storePotentialEdges(BitSet8 const& potMask, u32 edgeId, u32 parentNode)
{
	//size_t const count = potMask.count();
	u8 const bitMask = u8(potMask.to_ulong());

	if (bitMask == BITMASK_ALL_SET)
	{
		//store in current node
		storePotentialEdge(edgeId, parentNode, BITMASK_ALL_SET);
	}
	else
	{
		u32 const startingChild = octree->getChildrenStartingId(parentNode);

		for (u32 i = 0; i < OCTREE_NUM_CHILDREN; ++i)
		{
			if (potMask.test(i))
			{
				storePotentialEdge(edgeId, startingChild + i, BITMASK_ALL_SET);
			}
		}
	}
}

void CpuBuilder::processPotentialEdge(CurrentStatus const& status, u32 edgeId, BitSet8 const& potMask, Stack& stack)
{
	if(potMask.none())
	{
		return;
	}
	
	if (status.currentLevel == (octree->getDeepestLevel()-1))
	{
		if(IsCompressed)
		{
			storePotentialEdgesCompressed(potMask, edgeId, status.currentNode);
		}
		else
		{
			storePotentialEdges(potMask, edgeId, status.currentNode);
		}
	}
	else
	{
		pushPotNodesOnStack(potMask, status, stack);
	}
}

void CpuBuilder::storePotentialEdge(u32 edgeId, u32 nodeId, u8 bitmask)
{
#ifdef ENABLE_MULTITHREAD
	std::lock_guard<std::mutex> guard(Mutexes[nodeId * 2 * 256 + bitmask]);
#endif
	octree->getNode(nodeId)->edgesMayCastMap[bitmask].push_back(edgeId);
}

void CpuBuilder::storeSilhouetteEdge(u32 edgeId, u32 nodeId, u8 bitmask)
{
#ifdef ENABLE_MULTITHREAD
	std::lock_guard<std::mutex> guard(Mutexes[nodeId * 2 * 256 + 256 + bitmask]);
#endif
	octree->getNode(nodeId)->edgesAlwaysCastMap[bitmask].push_back(edgeId);
}

void CpuBuilder::pushPotNodesOnStack(BitSet8 const& mask, CurrentStatus const& status, Stack& stack)
{
	u8 const nextLevel = status.currentLevel + 1;
	u32 const firstChild = octree->getChildrenStartingId(status.currentNode);

	for (u32 i = 0; i < OCTREE_NUM_CHILDREN; ++i)
	{
		if (mask.test(i))
		{
			stack.push({ firstChild + i, nextLevel });
		}
	}
}

void CpuBuilder::propagatePotEdges()
{
	CpuPotEdgePropagator propagator;
	propagator.propagateEdgesToUpperLevels(octree, IsCompressed);
}

void CpuBuilder::createAllOctreeMasks()
{
	u32 const nofNodes = octree->getTotalNumNodes();
	for(u32 node = 0; node < nofNodes; ++node)
	{
		Node* n = octree->getNode(node);

		for(u8 i = 1; i<255; ++i)
		{
			n->edgesAlwaysCastMap[i].reserve(50);
			n->edgesMayCastMap[i].reserve(50);
		}
	}
}

void CpuBuilder::removeEmptyMasks()
{
	u32 const nofNodes = octree->getTotalNumNodes();
	for (u32 node = 0; node < nofNodes; ++node)
	{
		Node* n = octree->getNode(node);

		for (u8 i = 1; i < 255; ++i)
		{
			if(n->edgesAlwaysCastMap[i].size()==0)
			{
				n->edgesAlwaysCastMap.erase(i);
			}

			if(n->edgesMayCastMap[i].size()==0)
			{
				n->edgesMayCastMap.erase(i);
			}
		}
	}
}

void CpuBuilder::discardNodesWithLessThanEdges(u32 minNofEdges)
{
	u32 const nofNodes = octree->getTotalNumNodes();
	for (u32 node = 0; node < nofNodes; ++node)
	{
		Node* n = octree->getNode(node);
		s32 const startingChild = octree->getChildrenStartingId(node);

		if(startingChild<0)
		{
			continue; //all following nodes are 
		}

		for(auto mask = n->edgesAlwaysCastMap.begin(); mask!=n->edgesAlwaysCastMap.end(); )
		{
			if ((mask->first != BITMASK_ALL_SET) && (mask->second.size() < minNofEdges))
			{
				for (u32 bit = 0; bit < 8; ++bit)
				{
					if((mask->first>>bit) & 1)
					{
						Node* child = octree->getNode(startingChild + bit);
						child->edgesAlwaysCastMap[BITMASK_ALL_SET].insert(child->edgesAlwaysCastMap[BITMASK_ALL_SET].end(), mask->second.begin(), mask->second.end());
					}
				}

				//erase mask from parent
				n->edgesAlwaysCastMap.erase(mask++);
			}
			else
			{
				mask++;
			}
		}


		for (auto mask = n->edgesMayCastMap.begin(); mask != n->edgesMayCastMap.end(); )
		{
			if ((mask->first != BITMASK_ALL_SET) && (mask->second.size() < minNofEdges))
			{
				for (u32 bit = 0; bit < 8; ++bit)
				{
					if ((mask->first >> bit) & 1)
					{
						Node* child = octree->getNode(startingChild + bit);
						child->edgesMayCastMap[BITMASK_ALL_SET].insert(child->edgesMayCastMap[BITMASK_ALL_SET].end(), mask->second.begin(), mask->second.end());
					}
				}

				//erase mask from parent
				n->edgesMayCastMap.erase(mask++);
			}
			else
			{
				mask++;
			}
		}

	}
}

