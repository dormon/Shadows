#pragma once

#include<vector>
#include<limits>
#include<glm/glm.hpp>

class AABB{
  public:
    AABB(glm::vec3 const&mmin,glm::vec3 const&mmax):minVertex(mmin),maxVertex(mmax){}
    AABB():minVertex(std::numeric_limits<float>::max()),maxVertex(-std::numeric_limits<float>::max()){}
    AABB(std::vector<float>const&vertices):AABB(){
      for(size_t t=0;t<vertices.size();t+=3)
        addVertex(glm::vec3(vertices[t],vertices[t+1],vertices[t+2]));
    }
    glm::vec3 getMin()const{return minVertex;}
    glm::vec3 getMax()const{return maxVertex;}
    glm::vec3 getDiagonal()const{return maxVertex - minVertex;}
    bool      isValid()const{
      return glm::all(glm::greaterThanEqual(maxVertex,minVertex));
    }
    float     getVolume()const{
      auto const d = getDiagonal();
      return d.x * d.y * d.z;
    }
    AABB&     addVertex(glm::vec3 const&v){
      minVertex = glm::min(minVertex,v);
      maxVertex = glm::max(maxVertex,v);
      return*this;
    }
    AABB operator+(AABB const&b)const{
      AABB result;
      result.minVertex = glm::min(minVertex,b.getMin());
      result.maxVertex = glm::max(maxVertex,b.getMax());
      return result;
    }
    AABB operator*(AABB const&b)const{
      AABB result;
      result.minVertex = glm::max(minVertex,b.getMin());
      result.maxVertex = glm::min(maxVertex,b.getMax());
      return result;
    }
    AABB& setMin(glm::vec3 const&m){
      minVertex = m;
      return *this;
    }
    AABB& setMax(glm::vec3 const&m){
      maxVertex = m;
      return *this;
    }
    glm::vec3 getCenter()const{
      return(minVertex + maxVertex)/2.f;
    }
    AABB operator*(float s)const{
      auto const scaledHalfDiagonal = getDiagonal()/2.f*s;
      return AABB(
          getCenter() - scaledHalfDiagonal,
          getCenter() + scaledHalfDiagonal);
    }
  protected:
    glm::vec3 minVertex;
    glm::vec3 maxVertex;
};
