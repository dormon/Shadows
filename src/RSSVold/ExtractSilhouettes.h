#pragma once

#include<Vars/Vars.h>
#include<geGL/geGL.h>
#include<glm/glm.hpp>

namespace rssv{

class ExtractSilhouettes{
  public:
    ExtractSilhouettes(vars::Vars&vars);
    void extract(glm::vec4 const&light);
  protected:
    void createProgram();
    void createDispatchIndirectBuffer();
    void createEdgesBuffer();
    void createSilhouettesBuffer();
    vars::Vars&vars;
    std::shared_ptr<ge::gl::Program>program;
    std::shared_ptr<ge::gl::Buffer>edges;
    std::shared_ptr<ge::gl::Buffer>silhouettes;
    std::shared_ptr<ge::gl::Buffer>dispatchIndirect;
    size_t nofEdges = 0;
};

}
