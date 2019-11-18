#include <CSSV/createDIBO.h>
#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <FunctionPrologue.h>

using namespace ge::gl;

void cssv::createDIBO(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method");
  struct DrawArraysIndirectCommand{
    uint32_t nofVertices  = 0;
    uint32_t nofInstances = 0;
    uint32_t firstVertex  = 0;
    uint32_t baseInstance = 0;
  };
  DrawArraysIndirectCommand cmd;
  cmd.nofInstances = 1;
  vars.reCreate<Buffer>("cssv.method.dibo",sizeof(DrawArraysIndirectCommand),&cmd);
}

