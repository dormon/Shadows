#include <Vars/Vars.h>
#include <Model.h>
#include <FastAdjacency.h>
#include <AABB.h>

void getModelStats(vars::Vars&vars){
  auto model = vars.add<Model>("model",vars.getString("modelName"));
  std::vector<float>vertices = model->getVertices();
  auto adj = std::make_shared<Adjacency const>(vertices,vars.getSizeT("maxMultiplicity"));
  auto const aabb = AABB(vertices)*vars.getFloat("modelStatsScale");

  glm::uvec3 grid = *vars.get<glm::uvec3>("modelStatsGrid");
  size_t silhouetteCounter = 0;
  for(size_t z=0;z<grid.z;++z)
    for(size_t y=0;y<grid.y;++y)
      for(size_t x=0;x<grid.x;++x){
        std::cerr << x << "," << y << "," << z << std::endl;
        auto const light = aabb.getMin() + aabb.getDiagonal()/(glm::vec3(grid)-glm::vec3(1.f))*glm::vec3(x,y,z);
        //std::cerr << light.x << " " << light.y << " " << light.z << std::endl;
        for(size_t e=0;e<adj->getNofEdges();++e){
          auto const a = adj->getEdgeVertexA(e);
          auto const b = adj->getEdgeVertexB(e);
          auto const A = glm::vec3(vertices[a+0],vertices[a+1],vertices[a+2]);
          auto const B = glm::vec3(vertices[b+0],vertices[b+1],vertices[b+2]);
          auto const n = glm::normalize(glm::cross(light-A,B-A));
          auto const d = -glm::dot(A,n);
          int mult = 0;
          for(size_t o=0;o<adj->getNofOpposite(e);++o){
            auto const oi = adj->getOpposite(e,o);
            auto const O = glm::vec3(vertices[oi+0],vertices[oi+1],vertices[oi+2]);
            mult += -1+2*int(glm::dot(n,O) + d > 0);
          }
          if(mult != 0)silhouetteCounter++;
        }

      }


  float avgSil = static_cast<float>(silhouetteCounter) / static_cast<float>(grid.x*grid.y*grid.z);
  float triangles = (float)vertices.size() / 3.f ;
  float edges = (float)adj->getNofEdges();
  std::cerr << "nofTriangles,nofEdges,avgSil,silPerEdge,silPerTriangle" << std::endl;
  std::cerr << triangles << "," << edges << "," << avgSil << "," << avgSil/edges << "," << avgSil/triangles << std::endl;

  
  exit(0);
  }
