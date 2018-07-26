#pragma once

#include<Vars.h>
#include<glm/glm.hpp>
#include<geGL/geGL.h>

class Adjacency;
class CSSVExtractSilhouettes{
  public:
    CSSVExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
    virtual ~CSSVExtractSilhouettes(){}
    virtual void compute(glm::vec4 const&lightPosition);
    vars::Vars                      &vars        ;
    size_t                           nofEdges    ;
    std::shared_ptr<ge::gl::Buffer > edges       ;
    std::shared_ptr<ge::gl::Buffer > sillhouettes;
    std::shared_ptr<ge::gl::Buffer > dibo        ;
    std::shared_ptr<ge::gl::Program> program     ;
};

