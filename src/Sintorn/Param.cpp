#include<Sintorn/Param.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include<Vars/Caller.h>

void loadSintornParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars::Caller caller(vars,__FUNCTION__);
  vars.addUint32("args.sintorn.shadowFrustaPerWorkGroup") = arg->getu32("--sintorn-frustumsPerWorkgroup",1    ,"nof triangles solved by work group");
  vars.addFloat ("args.sintorn.bias"                    ) = arg->getf32("--sintorn-bias"                ,0.01f, "offset of triangle planes");
  vars.addBool  ("args.sintorn.discardBackFacing"       ) = arg->geti32("--sintorn-discardBackFacing"   ,1    ,"discard light back facing fragments from hierarchical depth ""texture construction");
  vars.addUint32("args.sintorn.shadowFrustaWGS"         ) = arg->getu32("--sintorn-shadowFrustaWGS"     ,64   ,"workgroups size of shadow frusta kernel");
}
