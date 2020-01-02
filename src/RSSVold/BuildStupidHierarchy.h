#pragma once

#include<RSSV/BuildHierarchy.h>
#include<geGL/geGL.h>

namespace rssv{

class BuildStupidHierarchy: public BuildHierarchy{
  public:
    BuildStupidHierarchy(vars::Vars&vars);
    virtual void build() override;
  protected:
    void allocateHierarchy();
    void createLevel0Program();
    void createNextLevelProgram();
    void copyLevel0();
    void buildNextLevel(size_t inputLevel);
    std::shared_ptr<ge::gl::Program>copyLevel0Program;
    std::shared_ptr<ge::gl::Program>buildNextLevelProgram;
    std::vector<std::shared_ptr<ge::gl::Buffer>>hierarchy;
    std::vector<size_t>nofPixelsPerLevel;
    std::vector<size_t>alignedNofPixelsPerLevel;
};

}
