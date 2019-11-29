#include <Sintorn2/param.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void sintorn2::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("sintorn2Param","parameters for Per-Triangle Shadow Volumes Using a View-Sample Cluster Hierarchy method");
  vars.addUint32("cssv.param.minZ"           ) = c->getu32   ("minZ"           ,0  ,"select number of Z bits - 0 mean max(xBits,yBits)"       );
}
