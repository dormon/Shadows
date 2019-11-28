#include <AdjacencyWrapper.h>
#include <FastAdjacency.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>

const float* getVertex3v(Adjacency const* ad, u32 vertexID)
{
	return &ad->getVertices()[0] + vertexID;
}

const glm::vec3& getEdgeVertexLow(Adjacency const* ad, u32 edgeId)
{
	return *(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getEdgeVertexA(edgeId)));
}

const glm::vec3& getEdgeVertexHigh(Adjacency const* ad, u32 edgeId)
{
	return *(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getEdgeVertexB(edgeId)));
}

u32 getNofOppositeVertices(Adjacency const* ad, u32 edgeId)
{
	return u32(ad->getNofOpposite(edgeId));
}

const glm::vec3& getOppositeVertex(Adjacency const* ad, u32 edgeId, u32 oppositeVertexID)
{
	return *(reinterpret_cast<const glm::vec3*>(getVertex3v(ad, u32(ad->getOpposite(size_t(edgeId), size_t(oppositeVertexID))))));
}

std::vector<glm::vec3> getEdgeOppositeVertices(Adjacency const* ad, u32 edgeId)
{
	const auto nofOpposite = getNofOppositeVertices(ad, edgeId);

	std::vector<glm::vec3> vertices;
	vertices.reserve(nofOpposite);

	for (u32 i = 0; i < nofOpposite; ++i)
	{
		vertices.push_back(*(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getOpposite(edgeId, i))));
	}

	return vertices;
}

