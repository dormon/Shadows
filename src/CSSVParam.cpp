#include <CSSVParam.h>
#include <ArgumentViewer/ArgumentViewer.h>

CSSVParams loadCSSVParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){

  CSSVParams cssvParams;

  cssvParams.computeSidesWGS = arg->getu32(
      "--cssv-WGS", 64, "compute silhouette shadow volumes work group size");
  cssvParams.localAtomic =
      arg->getu32("--cssv-localAtomic", 1, "use local atomic instructions");
  cssvParams.cullSides =
      arg->getu32("--cssv-cullSides", 0,
                  "enables culling of sides that are outside of viewfrustum");
  cssvParams.usePlanes =
      arg->getu32("--cssv-usePlanes", 0,
                  "use triangle planes instead of opposite vertices");
  cssvParams.useInterleaving =
      arg->getu32("--cssv-useInterleaving", 0,
                  "reorder edge that so it is struct of arrays");

  return cssvParams;
}
