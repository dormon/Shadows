#include <RSSV/param.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void rssv::loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  auto c = arg->getContext("rssvParam","parameters for ray traced silhouette shadow volumes");

  vars.addUint32("rssv.param.minZBits"               ) = c->getu32   ("minZBits"             ,10  ,"select number of Z bits - 0 mean max(xBits,yBits)"                               );
  vars.addUint32("rssv.param.propagateWarps"         ) = c->getu32   ("propagateWarps"       ,4   ,"number of warps cooperating on propagating data in hierarchy (for AMD 4 is good)");
  vars.addFloat ("rssv.param.bias"                   ) = c->getf32   ("bias"                 ,1.f ,"shadow frusta bias"                                                              );
  vars.addUint32("rssv.param.triangleAlignment"      ) = c->getu32   ("triangleAlignment"    ,128 ,"alignment of triangles"                                                          );
  vars.addUint32("rssv.param.sfAlignment"            ) = c->getu32   ("sfAlignment"          ,128 ,"shadow frusta alignment"                                                         );
  vars.addUint32("rssv.param.sfWGS"                  ) = c->getu32   ("sfWGS"                ,64  ,"shadow frusta work group size"                                                   );
  vars.addInt32 ("rssv.param.sfInterleave"           ) = c->geti32   ("sfInterleave"         ,1   ,"interleave shadow frusta floats"                                                 );
  vars.addInt32 ("rssv.param.triangleInterleave"     ) = c->geti32   ("triangleInterleave"   ,1   ,"interleave triangle floats"                                                      );
  vars.addInt32 ("rssv.param.morePlanes"             ) = c->geti32   ("morePlanes"           ,1   ,"additional frustum planes"                                                       );
  vars.addInt32 ("rssv.param.ffc"                    ) = c->geti32   ("ffc"                  ,1   ,"active front face culling"                                                       );
  vars.addInt32 ("rssv.param.noAABB"                 ) = c->geti32   ("noAABB"               ,0   ,"no tight aabb"                                                                   );
  vars.addSizeT ("rssv.param.alignment"              ) = c->getu64   ("alignment"            ,128 ,"buffer alignment in bytes"                                                       );
  vars.addUint32("rssv.param.extractSilhouettesWGS"  ) = c->getu32   ("extractSilhouettesWGS",64  ,"extract silhouettes work groups size"                                            );
  vars.addUint32("rssv.param.usePadding"             ) = c->getu32   ("usePadding"           ,1   ,"increase aabb size by half of a pixel"                                           );
  vars.addUint32("rssv.param.discardBackfacing"      ) = c->getu32   ("discardBackfacing"    ,1   ,"discard light backfacing fragments"                                              );
  vars.addInt32 ("rssv.param.memoryOptim"            ) = c->geti32   ("memoryOptim"          ,0   ,"apply memory optimization"                                                       );
  vars.addUint32("rssv.param.memoryFactor"           ) = c->getu32   ("memoryFactor"         ,10  ,"memory optimization - this value is average number of nodes per screen tile"     );
  vars.addUint32("rssv.param.useBridgePool"          ) = c->geti32   ("useBridgePool"        ,0   ,"create buffer containing information about bridges"                              );
  vars.addInt32 ("rssv.param.scaledQuantization"     ) = c->geti32   ("scaledQuantization"   ,0   ,"fix CPTSV quantization"                                                          );

  //FOR DEBUG
  vars.addBool  ("rssv.param.storeTraverseSilhouettesStat"  );
  vars.addBool  ("rssv.param.storeEdgePlanes"               );
  vars.addBool  ("rssv.param.dumpPointsNotPlanes"           );
  vars.addBool  ("rssv.param.computeBridges"                ,true);
  vars.addBool  ("rssv.param.storeBridgesInLocalMemory"     );
  vars.addBool  ("rssv.param.computeLastLevelSilhouettes"   ,true);

}
