#include <Vars/Vars.h>

#include <Methods.h>
#include <CubeShadowMapping/CubeShadowMapping.h>
#include <CSSV/CSSV.h>
#include <CSSVSOE.h>
#include <RSSV/RSSV.h>
#include <Sintorn/Sintorn.h>
#include <Sintorn2/Sintorn2.h>
#include <VSSV/VSSV.h>
#include <RayTracing/RayTracing.h>
#include <GSSV/GSSV.hpp>
#include <TSSV/TSSV.hpp>
#include <SM/ShadowMapping.h>
#include <DPM/DeepPrimitiveMap.h>
#include <ODPM/OmnidirDPM.h>
#include <DPSV/DPSV.h>
#include <HSSV/HSSV.h>
#include <MTSV/MTSV.h>

void initMethods(vars::Vars&vars){
  auto methods = vars.add<Methods>("methods");
  methods->add<CubeShadowMapping          >("cubeShadowMapping");
  methods->add<cssv::CSSV                 >("cssv"             );
  methods->add<CSSVSOE                    >("cssvsoe"          );
  methods->add<Sintorn                    >("sintorn"          );
  methods->add<Sintorn2                   >("sintorn2"         );
  methods->add<rssv::RSSV                 >("rssv"             );
  methods->add<VSSV                       >("vssv"             );
  methods->add<RayTracing                 >("rayTracing"       );
  methods->add<GSSV				          >("gssv"             );
  methods->add<TSSV				          >("tssv"             );
  methods->add<ShadowMapping	          >("sm"               );
  methods->add<DeepPrimitiveMap	          >("dpm"              );
  methods->add<OmnidirDpm                 >("odpm"             );
  methods->add<DPSV                       >("dpsv"             );
  methods->add<HSSV                       >("hssv"             );
  methods->add<MTSV                       >("mtsv"             );
}
