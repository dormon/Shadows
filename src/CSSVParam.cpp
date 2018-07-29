#include <CSSVParam.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadCSSVParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addUint32("cssv.computeSidesWGS") = arg->getu32   ("--cssv-WGS"            ,64,"compute silhouette shadow volumes work group size"       );
  vars.addUint32("cssv.localAtomic"    ) = arg->isPresent("--cssv-localAtomic"    ,   "use local atomic instructions"                              );
  vars.addBool  ("cssv.cullSides"      ) = arg->isPresent("--cssv-cullSides"      ,   "enables culling of sides that are outside of viewfrustum");
  vars.addBool  ("cssv.usePlanes"      ) = arg->isPresent("--cssv-usePlanes"      ,   "use triangle planes instead of opposite vertices"        );
  vars.addBool  ("cssv.useInterleaving") = arg->isPresent("--cssv-useInterleaving",   "reorder edge that so it is struct of arrays"             );
}
