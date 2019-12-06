#include <createAdjacency.h>
#include <FunctionPrologue.h>
#include <FastAdjacency.h>
#include <vector>
#include <Model.h>

using namespace std;

void createAdjacency(vars::Vars&vars){
  FUNCTION_PROLOGUE("","model","maxMultiplicity");

  vector<float> vertices = vars.get<Model>("model")->getVertices();

  //size_t const constexpr verticesPerTriangle = 3;
  //size_t const constexpr componentsPerVertex3D = 3;

  //size_t const nofTriangles = vertices.size() / (verticesPerTriangle*componentsPerVertex3D);

  vars.reCreate<Adjacency>("adjacency",vertices,vars.getSizeT("maxMultiplicity"));
}
