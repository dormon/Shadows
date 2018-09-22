#include<RSSV/Params.h>

#include<ArgumentViewer/ArgumentViewer.h>

void rssv::loadParams(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addSizeT ("rssv.extractSilhouetteWGS"   ) = arg->getu32("--rssv-computeSilhouettesWGS"  ,64,"workgroups size for silhouette computation");
  vars.addBool  ("rssv.localAtomic"            ) = arg->geti32("--rssv-localAtomic"            ,1 ,"use local atomic instructions in silhouette computation");
  vars.addBool  ("rssv.cullSides"              ) = arg->geti32("--rssv-cullSides"              ,0 ,"enables frustum culling of silhouettes");
  vars.addSizeT ("rssv.silhouettesPerWorkgroup") = arg->getu32("--rssv-silhouettesPerWorkgroup",1 ,"number of silhouette edges that are compute by one workgroup");
  vars.addBool  ("rssv.usePlanes"              ) = arg->geti32("--rssv-usePlanes"              ,0 ,"use triangle planes instead of opposite vertices");
  vars.addSizeT ("rssv.alignment"              ) = arg->getu32("--rssv-alignment"              ,1 ,"alignment of buffers");
  vars.add<glm::uvec2>("rssv.copyDepthToLastLevelOfHDTWGS",8,8);
}
