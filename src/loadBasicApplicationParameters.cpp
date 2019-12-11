#include<Vars/Vars.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include<FunctionPrologue.h>
#include<glm/glm.hpp>
#include<util.h>
#include<Methods.h>
#include<getMethodNameList.h>
#include<imguiVars/addVarsLimits.h>


void loadBasicApplicationParameters(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&args){
  vars::Caller caller(vars,__FUNCTION__);
  *vars.add<glm::uvec2 >("windowSize"     ) = vector2uvec2(args->getu32v("--window-size", {512, 512}, "window size"));

  *vars.add<glm::vec4  >("lightPosition"  ) = vector2vec4(args->getf32v("--light", {0.f, 1000.f, 0.f, 1.f}, "light position"));
  *vars.add<glm::vec3  >("lightUp") = vector2vec3(args->getf32v("--lightUp", { 1.f, 0.f, 0.f}, "light up vector (single-directional methods)"));
  *vars.add<glm::vec3  >("lightView") = vector2vec3(args->getf32v("--lightView", { 0.f, -1.f, 0.f}, "light view direction (single-directional methods)"));

  vars.addString        ("modelName"      ) = args->gets("--model", "/media/windata/ft/prace/models/2tri/2tri.3ds","model file name");
  vars.addBool          ("useShadows"     ) = !args->isPresent("--no-shadows", "turns off shadows");
  vars.addBool          ("verbose"        ) = args->isPresent("--verbose", "toggle verbose mode");

  vars.addString        ("methodName"     ) = args->gets("--method", "","name of shadow method: "+getMethodNameList(vars));
  vars.addSizeT         ("wavefrontSize"  ) = args->getu32("--wavefrontSize", 0,"warp/wavefront size, usually 32 for NVidia and 64 for AMD");
  vars.addSizeT         ("maxMultiplicity") = args->getu32("--maxMultiplicity", 2,"max number of triangles that share the same edge");
  vars.addBool          ("zfail"          ) = args->getu32("--zfail", 1, "shadow volumes zfail 0/1");
  vars.addBool          ("getModelStats"  ) = args->isPresent("--getModelStats","gets models stats - nof triangles, edges, silhouettes, ...");
  auto stats = args->getContext("modelStats","model stats parameters");
  *vars.add<glm::uvec3> ("modelStatsGrid" ) = vector2uvec3(stats->getu32v("grid",{10,10,10},"grid size"));
  vars.addFloat         ("modelStatsScale") = stats->getf32("scale",10.f,"scale factor");
  vars.addBool          ("dontTimeGbuffer") = args->isPresent("--dontTimeGbuffer", "GBuffer creation will not be timed");
}
