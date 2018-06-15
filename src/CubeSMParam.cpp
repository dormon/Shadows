#include<CubeSMParam.h>

#include<ArgumentViewer/ArgumentViewer.h>

void loadCubeShadowMappingParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addUint32("csm.resolution") = arg->getu32("--shadowMap-resolution", 1024  , "shadow map resolution"               );
  vars.addFloat ("csm.near"      ) = arg->getf32("--shadowMap-near"      , 0.1f  , "shadow map near plane position"      );
  vars.addFloat ("csm.far"       ) = arg->getf32("--shadowMap-far"       , 1000.f, "shadow map far plane position"       );
  vars.addUint32("csm.faces"     ) = arg->getu32("--shadowMap-faces"     , 6     , "number of used cube shadow map faces");
}

