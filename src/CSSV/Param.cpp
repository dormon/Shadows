#include <CSSV/Param.h>
#include <ArgumentViewer/ArgumentViewer.h>

void cssv::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("cssvParam","cssv parameters");
  vars.addUint32("cssv.param.computeSidesWGS") = c->getu32   ("WGS"            ,64,"compute silhouette shadow volumes work group size"       );
  vars.addBool  ("cssv.param.localAtomic"    ) = c->isPresent("localAtomic"    ,   "use local atomic instructions"                           );
  vars.addBool  ("cssv.param.cullSides"      ) = c->isPresent("cullSides"      ,   "enables culling of sides that are outside of viewfrustum");
  vars.addBool  ("cssv.param.usePlanes"      ) = c->isPresent("usePlanes"      ,   "use triangle planes instead of opposite vertices"        );
  vars.addBool  ("cssv.param.useInterleaving") = c->isPresent("useInterleaving",   "reorder edge that so it is struct of arrays"             );
  vars.addSizeT ("cssv.param.alignment"      ) = c->getu64   ("alignment"      ,1 ,"buffer alignment in bytes"                               );
}
