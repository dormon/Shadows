#include<VSSVParam.h>
#include<ArgumentViewer/ArgumentViewer.h>

void loadVSSVParams(
    vars::Vars &vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addBool("vssv.usePlanes"             ) = arg->geti32("--vssv-usePlanes"   ,0,"use planes instead of opposite vertices");
  vars.addBool("vssv.useStrips"             ) = arg->geti32("--vssv-useStrips"   ,1,"use triangle strips for sides of shadow volumes 0/1");
  vars.addBool("vssv.useAllOppositeVertices") = arg->geti32("--vssv-useAll"      ,0, "use all opposite vertices (even empty) 0/1");
  vars.addBool("vssv.drawCapsSeparately"    ) = arg->geti32("--vssv-capsSeparate",0, "draw caps using two draw calls");
}
