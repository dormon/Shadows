#pragma once

#include<cstdint>
#include<vector>
#include<iostream>
#include<glm/glm.hpp>

uint32_t getDispatchSize(std::size_t n,std::size_t wgs);
std::size_t divRoundUp(std::size_t a,std::size_t b);
glm::vec2 vector2vec2(std::vector<float> const& v);
glm::vec3 vector2vec3(std::vector<float> const& v);
glm::vec4 vector2vec4(std::vector<float> const& v);
glm::ivec2 vector2ivec2(std::vector<int32_t> const& v);
glm::ivec3 vector2ivec3(std::vector<int32_t> const& v);
glm::ivec4 vector2ivec4(std::vector<int32_t> const& v);
glm::uvec2 vector2uvec2(std::vector<uint32_t> const& v);
glm::uvec3 vector2uvec3(std::vector<uint32_t> const& v);
glm::uvec4 vector2uvec4(std::vector<uint32_t> const& v);
glm::vec2 vector2vec2(std::vector<double> const& v);
glm::vec3 vector2vec3(std::vector<double> const& v);
glm::vec4 vector2vec4(std::vector<double> const& v);
std::string uvec2ToStr(glm::uvec2 const& v);

size_t getWavefrontSize(size_t w=0);

template<typename RETURN,typename...ARGS>
class Barrier{
  public:
    using PTR = RETURN(*)(ARGS...);
    Barrier(PTR const&,RETURN && defRet,ARGS && ... defaults):returnValue(defRet),arguments{defaults...}{}
    bool notChanged(ARGS const&... args){
      auto newInputs = std::tuple<ARGS...>(args...);
      auto same = arguments == newInputs;
      if(same)return same;
      arguments = newInputs;
      return same;
    }
    RETURN             returnValue;
    std::tuple<ARGS...>arguments  ;
};

template<typename RETURN,typename...ARGS,typename VRET,typename...VARGS>
inline Barrier<RETURN,ARGS...>make_Barrier(RETURN(*ptr)(ARGS...),VRET && returnDef,VARGS && ...defaults){
  return Barrier<RETURN,ARGS...>{ptr,static_cast<RETURN>(returnDef),static_cast<ARGS>(defaults)...};
}

