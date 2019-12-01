#include <Sintorn2/param.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void sintorn2::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("sintorn2Param","parameters for Per-Triangle Shadow Volumes Using a View-Sample Cluster Hierarchy method");
  vars.addUint32("sintorn2.param.minZBits") = c->getu32("minZBits",9,"select number of Z bits - 0 mean max(xBits,yBits)");
  vars.addUint32("sintorn2.param.tileX"   ) = c->getu32("tileX"   ,8,"select tile X size"                               );
  vars.addUint32("sintorn2.param.tileY"   ) = c->getu32("tileY"   ,8,"select tile Y size"                               );

}