#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

class Adjacency {
 public:
  Adjacency(std::vector<float> const& vertices, size_t const& maxMultiplicity);
  size_t                    getNofEdges() const;
  size_t                    getEdge(size_t e, size_t i) const;
  size_t                    getEdgeVertexA(size_t e) const;
  size_t                    getEdgeVertexB(size_t e) const;
  size_t                    getNofOpposite(size_t e) const;
  size_t                    getOpposite(size_t e, size_t i) const;
  size_t                    getMaxMultiplicity() const;
  std::vector<float> const& getVertices() const;
  size_t                    getNofTriangles() const;

 protected:
  class EdgeAdjacency;
  std::vector<EdgeAdjacency> edges;
  std::vector<size_t> opposite;  ///< list of all indices to opposite vertices
  size_t              maxMultiplicity = 0;  ///< max allowed multiplicity
  std::vector<float>  vertices;             /// all vertices, with redundancy
};

class Adjacency::EdgeAdjacency {
 public:
  size_t ab[2];
  size_t offset;
  size_t count;
  EdgeAdjacency(size_t const& a,
                size_t const& b,
                size_t const& o,
                size_t const& c)
  {
    ab[0]  = a;
    ab[1]  = b;
    offset = o;
    count  = c;
  }
};
