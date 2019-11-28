#include <Node.h>
#include <algorithm>

bool Node::isValid() const
{
	return volume.isValid();
}

void Node::clear()
{
	edgesMayCastMap.clear();
	edgesAlwaysCastMap.clear();
	volume = AABB();
}

void Node::shrinkEdgeVectors()
{
	for (auto e : edgesMayCastMap)
		e.second.shrink_to_fit();

	for (auto e : edgesAlwaysCastMap)
		e.second.shrink_to_fit();
}

void Node::sortEdgeVectors()
{
	for (auto e : edgesMayCastMap)
		std::sort(e.second.begin(), e.second.end());

	for (auto e : edgesAlwaysCastMap)
		std::sort(e.second.begin(), e.second.end());
}