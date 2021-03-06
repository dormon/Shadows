#pragma once

#include <Defines.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

class Adjacency;

const float* getVertex3v(Adjacency const* ad, u32 vertexID);

const glm::vec3& getEdgeVertexLow(Adjacency const* ad, u32 edgeId);

const glm::vec3& getEdgeVertexHigh(Adjacency const* ad, u32 edgeId);

u32 getNofOppositeVertices(Adjacency const* ad, u32 edgeId);

const glm::vec3& getOppositeVertex(Adjacency const* ad, u32 edgeId, u32 oppositeVertexID);

std::vector<glm::vec3> getEdgeOppositeVertices(Adjacency const* ad, u32 edgeId);

