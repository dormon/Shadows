#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

class Adjacency;

const float* getVertex3v(Adjacency const* ad, size_t vertexID);

const glm::vec3& getEdgeVertexLow(Adjacency const* ad, size_t edgeId);

const glm::vec3& getEdgeVertexHigh(Adjacency const* ad, size_t edgeId);

size_t getNofOppositeVertices(Adjacency const* ad, size_t edgeId);

const glm::vec3& getOppositeVertex(Adjacency const* ad, size_t edgeId, size_t oppositeVertexID);

std::vector<glm::vec3> getEdgeOppositeVertices(Adjacency const* ad, size_t edgeId);

