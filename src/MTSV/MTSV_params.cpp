#include "MTSV_params.h"
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadMtsvParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("mtsv.args.wgSize") = arg->getu32("--mtsv-wgSize", 512, "Workgroup size");
	vars.addUint32("mtsv.args.numWg") = arg->getu32("--mtsv-numWg", 32, "MTSV nof work groups for compute shader persistent threads");
	vars.addFloat("mtsv.args.bias") = arg->getf32("--mtsv-bias", 0.f, "MTSV bias");
	vars.addBool("mtsv.args.useFrontFaceCulling") = arg->isPresent("--mtsv-useFfCulling", "Uses front face culling during build");
}

