#include<CubeShadowMapping/Params.h>

#include<ArgumentViewer/ArgumentViewer.h>

void loadShadowMappingParams( vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg) 
{
	vars.addUint32("args.sm.resolution")     = arg->getu32("--shadowMap-resolution", 1024, "shadow map resolution");
	vars.addFloat ("args.sm.near")           = arg->getf32("--shadowMap-near", 0.1f, "shadow map near plane position");
	vars.addFloat ("args.sm.far")            = arg->getf32("--shadowMap-far", 1000.f, "shadow map far plane position");
	vars.addFloat ("args.sm.fovy")           = arg->getf32("--shadowMap-fovy", 1.5707963267948966f, "shadow map fovY");
	vars.addUint32("args.sm.pcf")            = arg->getu32("--shadowMap-pcf", 0, "shadow map PCF kernel size");
}