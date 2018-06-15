#include<SintornParam.h>
#include<ArgumentViewer/ArgumentViewer.h>

SintornParams loadSintornParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  SintornParams sintornParams;
  sintornParams.shadowFrustaPerWorkGroup =
      arg->geti32("--sintorn-frustumsPerWorkgroup", 1,
                  "nof triangles solved by work group");
  sintornParams.bias =
      arg->getf32("--sintorn-bias", 0.01f, "offset of triangle planes");
  sintornParams.discardBackFacing =
      arg->geti32("--sintorn-discardBackFacing", 1,
                  "discard light back facing fragments from hierarchical depth "
                  "texture construction");
  return sintornParams;
}
