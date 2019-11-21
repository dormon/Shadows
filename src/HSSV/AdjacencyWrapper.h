#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

class Adjacency;

const float* getVertex3v(Adjacency* ad, size_t vertexID);

const glm::vec3& getEdgeVertexLow(Adjacency* ad, size_t edgeId);

const glm::vec3& getEdgeVertexHigh(Adjacency* ad, size_t edgeId);

size_t getNofOppositeVertices(Adjacency* ad, size_t edgeId);

const glm::vec3& getOppositeVertex(Adjacency* ad, size_t edgeId, size_t oppositeVertexID);

std::vector<glm::vec3> getEdgeOppositeVertices(Adjacency* ad, size_t edgeId);

