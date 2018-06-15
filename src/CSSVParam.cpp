#include <CSSVParam.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadCSSVParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addUint32("cssv.computeSidesWGS") = arg->getu32("--cssv-WGS"            ,64,"compute silhouette shadow volumes work group size"       );
  vars.addUint32("cssv.localAtomic"    ) = arg->getu32("--cssv-localAtomic"    ,1 ,"use local atomic instructions"                           );
  vars.addBool  ("cssv.cullSides"      ) = arg->getu32("--cssv-cullSides"      ,0 ,"enables culling of sides that are outside of viewfrustum");
  vars.addBool  ("cssv.usePlanes"      ) = arg->getu32("--cssv-usePlanes"      ,0 ,"use triangle planes instead of opposite vertices"        );
  vars.addBool  ("cssv.useInterleaving") = arg->getu32("--cssv-useInterleaving",0 ,"reorder edge that so it is struct of arrays"             );
}
