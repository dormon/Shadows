#include<RSSVParam.h>

#include<ArgumentViewer/ArgumentViewer.h>

RSSVParams loadRSSVParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  RSSVParams rssvParams;
  rssvParams.computeSilhouetteWGS =
      arg->geti32("--rssv-computeSilhouettesWGS", 64,
                  "workgroups size for silhouette computation");
  rssvParams.localAtomic =
      arg->geti32("--rssv-localAtomic", 1,
                  "use local atomic instructions in silhouette computation");
  rssvParams.cullSides               = arg->geti32("--rssv-cullSides", 0,
                                     "enables frustum culling of silhouettes");
  rssvParams.silhouettesPerWorkgroup = arg->geti32(
      "--rssv-silhouettesPerWorkgroup", 1,
      "number of silhouette edges that are compute by one workgroup");
  rssvParams.usePlanes =
      arg->geti32("--rssv-usePlanes", 0,
                  "use triangle planes instead of opposite vertices");
  return rssvParams;
}
