#include <CSSV/Param.h>
#include <ArgumentViewer/ArgumentViewer.h>

void cssv::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("cssvParam","cssv parameters");
  vars.addUint32("cssv.param.computeSidesWGS"        ) = c->getu32   ("WGS"                    ,64 ,"compute silhouette shadow volumes work group size"       );
  vars.addBool  ("cssv.param.noLocalAtomic"          ) = c->isPresent("noLocalAtomic"          ,    "dont use local atomic instructions"                      );
  vars.addBool  ("cssv.param.cullSides"              ) = c->isPresent("cullSides"              ,    "enables culling of sides that are outside of viewfrustum");
  vars.addBool  ("cssv.param.dontUsePlanes"          ) = c->isPresent("dontUsePlanes"          ,    "dont use triangle planes instead of opposite vertices"   );
  vars.addBool  ("cssv.param.dontUseInterleaving"    ) = c->isPresent("dontUseInterleaving"    ,    "dont reorder edge that so it is struct of arrays"        );
  vars.addBool  ("cssv.param.dontExtractMultiplicity") = c->isPresent("dontExtractMultiplicity",    "dont extract only multiplicity"                               );
  vars.addBool  ("cssv.param.dontPackMult"           ) = c->isPresent("dontPackMult"           ,    "dont pack multiplicity sign and edge id"                 );
  vars.addSizeT ("cssv.param.alignment"              ) = c->getu64   ("alignment"              ,128,"buffer alignment in bytes"                               );
}
