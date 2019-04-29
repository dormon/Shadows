#include <Vars/Vars.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <CubeShadowMapping/Params.h>
#include <CSSV/Param.h>
#include <VSSV/Params.h>
#include <Sintorn/Param.h>
#include <RSSV/Params.h>
#include <loadBasicApplicationParameters.h>
#include <loadTestParams.h>
#include <loadCameraParams.h>

void parseArguments(vars::Vars&vars){
  auto argc = vars.getUint32("argc");
  auto argv = *vars.get<char**>("argv");
  auto arg = std::make_shared<argumentViewer::ArgumentViewer>(argc, argv);

  loadBasicApplicationParameters(vars,arg);
  loadCubeShadowMappingParams   (vars,arg);
  cssv::loadParams              (vars,arg);
  loadVSSVParams                (vars,arg);
  loadSintornParams             (vars,arg);
  rssv::loadParams              (vars,arg);
  loadTestParams                (vars,arg);
  loadCameraParams              (vars,arg);

  vars.addSizeT("cssvsoe.computeSidesWGS") = arg->getu32(
      "--cssvsoe-WGS", 64, "compute silhouette shadow volumes work group size");

  vars.addSizeT("frameCounter");
  vars.addSizeT("maxFrame") = arg->getu32("--maxFrame",0,"after this frame the app will stop");


  bool printHelp = arg->isPresent("-h", "prints this help");
  if (printHelp || !arg->validate()) {
    std::cerr << arg->toStr();
    exit(0);
  }

}
