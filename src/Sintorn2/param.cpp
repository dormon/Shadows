#include <Sintorn2/param.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void sintorn2::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("sintorn2Param","parameters for Per-Triangle Shadow Volumes Using a View-Sample Cluster Hierarchy method");

  vars.addUint32("sintorn2.param.minZBits"               ) = c->getu32   ("minZBits"          ,9   ,"select number of Z bits - 0 mean max(xBits,yBits)"                               );
  vars.addUint32("sintorn2.param.tileX"                  ) = c->getu32   ("tileX"             ,8   ,"select tile X size"                                                              );
  vars.addUint32("sintorn2.param.tileY"                  ) = c->getu32   ("tileY"             ,8   ,"select tile Y size"                                                              );
  vars.addUint32("sintorn2.param.propagateWarps"         ) = c->getu32   ("propagateWarps"    ,4   ,"number of warps cooperating on propagating data in hierarchy (for AMD 4 is good)");
  vars.addFloat ("sintorn2.param.bias"                   ) = c->getf32   ("bias"              ,0.1f,"shadow frusta bias"                                                              );
  vars.addUint32("sintorn2.param.triangleAlignment"      ) = c->getu32   ("triangleAlignment" ,128 ,"alignment of triangles"                                                          );
  vars.addUint32("sintorn2.param.sfAlignment"            ) = c->getu32   ("sfAlignment"       ,128 ,"shadow frusta alignment"                                                         );
  vars.addUint32("sintorn2.param.sfWGS"                  ) = c->getu32   ("sfWGS"             ,64  ,"shadow frusta work group size"                                                   );
  vars.addInt32 ("sintorn2.param.sfInterleave"           ) = c->geti32   ("sfInterleave"      ,1   ,"interleave shadow frusta floats"                                                 );
  vars.addInt32 ("sintorn2.param.triangleInterleave"     ) = c->geti32   ("triangleInterleave",1   ,"interleave triangle floats"                                                      );
  vars.addInt32 ("sintorn2.param.morePlanes"             ) = c->geti32   ("morePlanes"        ,1   ,"additional frustum planes"                                                       );

}
