#include<ShadowMethod.h>

namespace rssv{

class RSSV: public ShadowMethod{
  public:
    RSSV(vars::Vars&vars);
    virtual ~RSSV();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawDebug(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
};

}
