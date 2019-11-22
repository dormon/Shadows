#include <AdjacencyWrapper.h>
#include <FastAdjacency.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>

const float* getVertex3v(Adjacency const* ad, size_t vertexID)
{
	return &ad->getVertices()[0] + vertexID;
}

const glm::vec3& getEdgeVertexLow(Adjacency const* ad, size_t edgeId)
{
	return *(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getEdgeVertexA(edgeId)));
}

const glm::vec3& getEdgeVertexHigh(Adjacency const* ad, size_t edgeId)
{
	return *(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getEdgeVertexB(edgeId)));
}

size_t getNofOppositeVertices(Adjacency const* ad, size_t edgeId)
{
	return ad->getNofOpposite(edgeId);
}

const glm::vec3& getOppositeVertex(Adjacency const* ad, size_t edgeId, size_t oppositeVertexID)
{
	return *(reinterpret_cast<const glm::vec3*>(getVertex3v(ad, ad->getOpposite(edgeId, oppositeVertexID))));
}

std::vector<glm::vec3> getEdgeOppositeVertices(Adjacency const* ad, size_t edgeId)
{
	const auto nofOpposite = getNofOppositeVertices(ad, edgeId);

	std::vector<glm::vec3> vertices;
	vertices.reserve(nofOpposite);

	for (uint32_t i = 0; i < nofOpposite; ++i)
	{
		vertices.push_back(*(reinterpret_cast<const glm::vec3*>(&ad->getVertices()[0] + ad->getOpposite(edgeId, i))));
	}

	return vertices;
}

