#include <Sintorn2/param.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void sintorn2::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("sintorn2Param","parameters for Per-Triangle Shadow Volumes Using a View-Sample Cluster Hierarchy method");

  vars.addUint32("sintorn2.param.minZBits"                      ) =       c->getu32 ("minZBits"                      ,10  ,"select number of Z bits - 0 mean max(xBits,yBits)"                               );
  vars.addUint32("sintorn2.param.tileX"                         ) =       c->getu32 ("tileX"                         ,8   ,"select tile X size"                                                              );
  vars.addUint32("sintorn2.param.tileY"                         ) =       c->getu32 ("tileY"                         ,8   ,"select tile Y size"                                                              );
  vars.addUint32("sintorn2.param.propagateWarps"                ) =       c->getu32 ("propagateWarps"                ,4   ,"number of warps cooperating on propagating data in hierarchy (for AMD 4 is good)");
  vars.addFloat ("sintorn2.param.bias"                          ) =       c->getf32 ("bias"                          ,1.f ,"shadow frusta bias"                                                              );
  vars.addUint32("sintorn2.param.triangleAlignment"             ) =       c->getu32 ("triangleAlignment"             ,128 ,"alignment of triangles"                                                          );
  vars.addUint32("sintorn2.param.sfAlignment"                   ) =       c->getu32 ("sfAlignment"                   ,128 ,"shadow frusta alignment"                                                         );
  vars.addUint32("sintorn2.param.sfWGS"                         ) =       c->getu32 ("sfWGS"                         ,64  ,"shadow frusta work group size"                                                   );
  vars.addBool  ("sintorn2.param.sfInterleave"                  ) = (bool)c->geti32 ("sfInterleave"                  ,1   ,"interleave shadow frusta floats"                                                 );
  vars.addBool  ("sintorn2.param.triangleInterleave"            ) = (bool)c->geti32 ("triangleInterleave"            ,1   ,"interleave triangle floats"                                                      );
  vars.addBool  ("sintorn2.param.morePlanes"                    ) = (bool)c->geti32 ("morePlanes"                    ,1   ,"additional frustum planes"                                                       );
  vars.addBool  ("sintorn2.param.ffc"                           ) = (bool)c->geti32 ("ffc"                           ,1   ,"active front face culling"                                                       );
  vars.addBool  ("sintorn2.param.noAABB"                        ) = (bool)c->geti32 ("noAABB"                        ,0   ,"no tight aabb"                                                                   );
  vars.addBool  ("sintorn2.param.memoryOptim"                   ) = (bool)c->geti32 ("memoryOptim"                   ,1   ,"apply memory optimization"                                                       );
  vars.addUint32("sintorn2.param.memoryFactor"                  ) =       c->getu32 ("memoryFactor"                  ,10  ,"memory optimization - this value is average number of nodes per screen tile"     );
  vars.addBool  ("sintorn2.param.taOptim"                       ) = (bool)c->geti32 ("taOptim"                       ,1   ,"apply trivial accept optim. that erases parts of already shadowed tree"          );
  vars.addBool  ("sintorn2.param.triangleIntersect"             ) = (bool)c->geti32 ("triangleIntersect"             ,0   ,"debug only, converts shadow frustum to triangle during intersection"             );
  vars.addBool  ("sintorn2.param.discardBackfacing"             ) = (bool)c->geti32 ("discardBackfacing"             ,1   ,"discard light backfacing fragments"                                              );
  vars.addBool  ("sintorn2.param.computeLastLevel"              ) = (bool)c->geti32 ("computeLastLevel"              ,1   ,"computes last level of hierarchy during raterization"                            );
  vars.addFloat ("sintorn2.param.lightTriangleDistanceThreshold") =       c->getf32 ("lightTriangleDisntaceThreshold",500.f,"if the triangle light distance is less that this, \"more planes\" are discarded");


  vars.addBool  ("sintorn2.param.storeTraverseStat");

}
