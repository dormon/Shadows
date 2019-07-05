#include<ShadowMethod.h>

class Sintorn2: public ShadowMethod{
  public:
    Sintorn2(vars::Vars&vars);
    virtual ~Sintorn2();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
};
