#pragma once

#include <geGL/geGL.h>
#include <glm/glm.hpp>
#include <ShadowVolumes.h>
#include <Model.h>
#include <TimeStamp.h>
#include <Vars/Vars.h>
#include <CSSV/Param.h>
#include <CSSV/ExtractSilhouettes.h>

class Adjacency;

class cssv::CSSV: public ShadowVolumes{
  public:
    CSSV(vars::Vars&vars);
    virtual ~CSSV();
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    std::shared_ptr<ge::gl::Program>       _drawSidesProgram  = nullptr;
    std::unique_ptr<cssv::ExtractSilhouettes>extractSilhouettes          ;
    std::shared_ptr<ge::gl::VertexArray>   _sidesVao                   ;
    size_t                                 _nofTriangles      = 0      ;
    std::shared_ptr<ge::gl::Program>       _drawCapsProgram   = nullptr;
    std::shared_ptr<ge::gl::VertexArray>   _capsVao           = nullptr;
    std::shared_ptr<ge::gl::Buffer>        _caps              = nullptr;
    void _createCapsData(std::shared_ptr<Adjacency const>const&adj);
};

