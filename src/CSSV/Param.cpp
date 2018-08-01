#include <CSSV/Param.h>
#include <ArgumentViewer/ArgumentViewer.h>

void cssv::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("cssvParam","cssv parameters");

  vars.addUint32("cssv.computeSidesWGS") = c->getu32   ("WGS"            ,64,"compute silhouette shadow volumes work group size"       );
  vars.addUint32("cssv.localAtomic"    ) = c->isPresent("localAtomic"    ,   "use local atomic instructions"                           );
  vars.addBool  ("cssv.cullSides"      ) = c->isPresent("cullSides"      ,   "enables culling of sides that are outside of viewfrustum");
  vars.addBool  ("cssv.usePlanes"      ) = c->isPresent("usePlanes"      ,   "use triangle planes instead of opposite vertices"        );
  vars.addBool  ("cssv.useInterleaving") = c->isPresent("useInterleaving",   "reorder edge that so it is struct of arrays"             );
  vars.addSizeT ("cssv.alignment"      ) = c->getu64   ("alignment"      ,1 ,"buffer alignment in bytes"                               );
}
