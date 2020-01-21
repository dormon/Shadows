#include <numeric>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <RSSV/allocateHierarchy.h>
#include <RSSV/config.h>

using namespace ge::gl;

void rssv::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method","rssv.method.config");

  auto cfg = *vars.get<Config>("rssv.method.config");

  vars.reCreate<Buffer >("rssv.method.nodePool"        ,cfg.nodesSize                                  );
  vars.reCreate<Buffer >("rssv.method.aabbPool"        ,cfg.aabbsSize                                  );
  vars.reCreate<Buffer >("rssv.method.levelNodeCounter",cfg.nofLevels*sizeof(uint32_t)*4               );
  vars.reCreate<Buffer >("rssv.method.activeNodes"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.debugBuffer"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.bridges"         ,cfg.nofNodes *sizeof(int32_t)                  );
  vars.reCreate<Texture>("rssv.method.stencil"         ,GL_TEXTURE_2D,GL_R32I,1,cfg.windowX,cfg.windowY);
}
