#include <Vars/Vars.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <CubeShadowMapping/Params.h>
#include <CSSV/Param.h>
#include <VSSV/Params.h>
#include <Sintorn/Param.h>
#include <RSSV/param.h>
#include <loadBasicApplicationParameters.h>
#include <loadTestParams.h>
#include <loadCameraParams.h>
#include <imguiVars/addVarsLimits.h>
#include <GSSV/GSSV_params.h>
#include <TSSV/TSSV_Params.h>
#include <SM/ShadowMappingParams.h>
#include <DPM/DpmParams.h>
#include <ODPM/OdpmParams.h>
#include <DPSV/DpsvParams.h>
#include <HSSV/HssvParams.h>
#include <Sintorn2/param.h>
#include <MTSV/MTSV_params.h>
#include <FTS/FTS_params.h>
#include <DPSM/DPSM_params.h>

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
  loadGSSVParams				(vars,arg);
  loadTSSVParams                (vars,arg);
  loadShadowMappingParams       (vars,arg);
  loadDpmParams                 (vars,arg);
  loadOdpmParams                (vars,arg);
  loadDpsvParams                (vars,arg);
  loadHssvParams                (vars,arg);
  sintorn2::loadParams          (vars,arg);
  loadMtsvParams                (vars,arg);
  loadFtsParams                 (vars,arg);
  loadDpsmParams                (vars, arg);

  vars.addSizeT("cssvsoe.computeSidesWGS") = arg->getu32(
      "--cssvsoe-WGS", 64, "compute silhouette shadow volumes work group size");

  vars.addBool("notResizable") = arg->isPresent("--notResizable");

  vars.addSizeT("frameCounter");
  vars.addSizeT("maxFrame") = arg->getu32("--maxFrame",0,"after this frame the app will stop");
  vars.addBool("args.camera.remember") = arg->isPresent("--camera-remember","if present it will load camera at start and store camera at the end");

  hide(vars,"frameCounter");
  hide(vars,"maxFrame");


  bool printHelp = arg->isPresent("-h", "prints this help");
  if (printHelp || !arg->validate()) {
    std::cerr << arg->toStr();
    exit(0);
  }

}
