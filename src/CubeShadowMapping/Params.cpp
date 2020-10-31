#include<CubeShadowMapping/Params.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadCubeShadowMappingParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addUint32("csm.param.resolution") = arg->getu32("--shadowMap-resolution", 1024  , "shadow map resolution"               );
  vars.addFloat ("csm.param.near"      ) = arg->getf32("--shadowMap-near"      , 0.1f  , "shadow map near plane position"      );
  vars.addFloat ("csm.param.far"       ) = arg->getf32("--shadowMap-far"       , 1000.f, "shadow map far plane position"       );
  vars.addUint32("csm.param.faces"     ) = arg->getu32("--shadowMap-faces"     , 6     , "number of used cube shadow map faces");
  vars.addFloat ("csm.param.factor"    ) = arg->getf32("--shadowMap-factor"    , 2.5f  , "factor in polygon offset"            );
  vars.addFloat ("csm.param.units"     ) = arg->getf32("--shadowMap-units"     , 10.f  , "units in polygon offset"             );
}

