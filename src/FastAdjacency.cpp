#include <FastAdjacency.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace std;

template <size_t dim>
class Vertex {
 public:
  const float* data  = nullptr;  ///< global data
  size_t       index = 0;        ///< this vertex starts here
  Vertex(const float* d = nullptr, size_t i = 0):data(d),index(i){}
  int compare(Vertex<dim> const& b) const
  {
    for (size_t d = 0; d < dim; ++d) {
      if (data[index + d] > b.data[b.index + d]) return 1;
      if (data[index + d] < b.data[b.index + d]) return -1;
    }
    return 0;
  }
  string toStr() const
  {
    stringstream ss;
    for (size_t d = 0; d < dim; ++d) {
      ss << data[index + d];
      if (d < dim - 1) ss << " ";
    }
    return ss.str();
  }
};

class EdgeToTriangle {
 public:
  Vertex<3> a;
  Vertex<3> b;
  Vertex<3> c;
  EdgeToTriangle(Vertex<3> const& aa, Vertex<3> const& bb, Vertex<3> const& cc)
  {
    a = aa;
    b = bb;
    c = cc;
  }
  bool operator<(EdgeToTriangle const& edge) const
  {
    int aea = a.compare(edge.a);
    if (aea < 0) return true;
    if (aea > 0) return false;
    int beb = b.compare(edge.b);
    if (beb < 0) return true;
    if (beb > 0) return false;
    int cec = c.compare(edge.c);
    if (cec < 0) return true;
    if (cec > 0) return false;
    return false;
  }
  bool edgeEqual(EdgeToTriangle const& edge)
  {
    if (a.compare(edge.a) != 0) return false;
    if (b.compare(edge.b) != 0) return false;
    return true;
  }
  bool operator==(EdgeToTriangle const& edge) const
  {
    if (a.compare(edge.a) != 0) return false;
    if (b.compare(edge.b) != 0) return false;
    if (c.compare(edge.c) != 0) return false;
    return true;
  }
};

/**
 * @brief constructor of adjacency information
 *
 * @param verts three 3D consecutive vertices form a triangle
 * @param maxMultiplicity edges with larger nof opposite vertices will be split
 */
Adjacency::Adjacency(vector<float> const& verts, size_t const& maxMult)
{
  vertices                            = verts;
  maxMultiplicity                     = maxMult;
  auto const             nofTriangles = vertices.size() / 3 / 3;
  vector<EdgeToTriangle> edgeToTriangle;
  edgeToTriangle.reserve(3 * nofTriangles);
  for (size_t t = 0; t < nofTriangles; ++t) {  // loop over triangles
    Vertex<3> a(vertices.data(), (t * 3 + 0) * 3);
    Vertex<3> b(vertices.data(), (t * 3 + 1) * 3);
    Vertex<3> c(vertices.data(), (t * 3 + 2) * 3);
    int       ab = a.compare(b);
    int       ac = a.compare(c);
    int       bc = b.compare(c);
    if (ab == 0) continue;
    if (ac == 0) continue;
    if (bc == 0) continue;
    if (ab < 0)
      edgeToTriangle.push_back(EdgeToTriangle(a, b, c));
    else
      edgeToTriangle.push_back(EdgeToTriangle(b, a, c));
    if (ac < 0)
      edgeToTriangle.push_back(EdgeToTriangle(a, c, b));
    else
      edgeToTriangle.push_back(EdgeToTriangle(c, a, b));
    if (bc < 0)
      edgeToTriangle.push_back(EdgeToTriangle(b, c, a));
    else
      edgeToTriangle.push_back(EdgeToTriangle(c, b, a));
  }
  sort(edgeToTriangle.begin(), edgeToTriangle.end());

  edges.push_back(EdgeAdjacency(edgeToTriangle[0].a.index,
                                edgeToTriangle[0].b.index, 0, 1));
  opposite.push_back(edgeToTriangle[0].c.index);

  size_t uniqueIndex = 0;
  for (size_t i = 1; i < edgeToTriangle.size(); ++i) {
    if ((--edges.end())->count < maxMultiplicity &&
        edgeToTriangle[uniqueIndex].edgeEqual(edgeToTriangle[i])) {
      opposite.push_back(edgeToTriangle[i].c.index);
      (--edges.end())->count++;
      continue;
    }

    size_t offset = (--edges.end())->count + (--edges.end())->offset;
    edges.push_back(EdgeAdjacency(edgeToTriangle[i].a.index,
                                  edgeToTriangle[i].b.index, offset, 1));
    opposite.push_back(edgeToTriangle[i].c.index);
    uniqueIndex = i;
  }
}

/**
 * @brief gets number of edges
 *
 * @return
 */
size_t Adjacency::getNofEdges() const { return edges.size(); }

/**
 * @brief gets indices of edge vertices
 *
 * @param e edge number
 * @param i 0 - vertex a, 1 - vertex b of edge e
 *
 * @return index of vertex a or b of edge e
 */
size_t Adjacency::getEdge(size_t e, size_t i) const
{
  assert(i < 2);
  return edges[e].ab[i];
}

/**
 * @brief gets index of edge vertex A
 *
 * @param e edge index
 *
 * @return index of vertex A of an edge e
 */
size_t Adjacency::getEdgeVertexA(size_t e) const { return getEdge(e, 0); }

/**
 * @brief gets index of vertex B of edge
 *
 * @param e
 *
 * @return
 */
size_t Adjacency::getEdgeVertexB(size_t e) const { return getEdge(e, 1); }

/**
 * @brief gets number of opposite vertices of edge e
 *
 * @param e edge number
 *
 * @return number of opposite vertice of edge e
 */
size_t Adjacency::getNofOpposite(size_t e) const { return edges[e].count; }

/**
 * @brief gets index of opposite vertex
 *
 * @param e edge e
 * @param i ith opposite vertex
 *
 * @return index of ith opposite vertex of edge e
 */
size_t Adjacency::getOpposite(size_t e, size_t i) const
{
  return opposite[edges[e].offset + i];
}

/**
 * @brief gets maximal multiplicity
 *
 * @return maximal multiplicity
 */
size_t Adjacency::getMaxMultiplicity() const { return maxMultiplicity; }

/**
 * @brief gets array of vertices
 *
 * @return array of vertices
 */
vector<float> const& Adjacency::getVertices() const { return vertices; }

/**
 * @brief gets nof triangles
 *
 * @return number of triangles
 */
size_t Adjacency::getNofTriangles() const { return vertices.size() / 3 / 3; }
