#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <BallotShader.h>
#include <Deferred.h>
#include <FastAdjacency.h>

#include <RSSV/rasterize.h>
#include <RSSV/rasterizeShader.h>
#include <RSSV/getConfigShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/quantizeZShader.h>
#include <RSSV/depthToZShader.h>
#include <RSSV/collisionShader.h>
#include <RSSV/getEdgePlanesShader.h>
#include <RSSV/traverseSilhouettesShader.h>
#include <RSSV/traverseTrianglesShader.h>
#include <RSSV/traverseShader.h>
#include <RSSV/getAABBShader.h>
#include <RSSV/loadEdgeShader.h>
#include <RSSV/sharedMemoryShader.h>
#include <RSSV/mergeShader.h>
#include <RSSV/globalBarrierShader.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>
#include <RSSV/config.h>
#include <perfCounters.h>

using namespace ge::gl;
using namespace std;

namespace rssv{

void createTraverseProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"adjacency"
      ,"rssv.method.config"
      ,"rssv.param.sfWGS"
      ,"rssv.param.bias"
      ,"rssv.param.noAABB"

      //debug
      ,"rssv.param.storeTraverseSilhouettesStat"
      ,"rssv.param.storeTraverseTrianglesStat"
      ,"rssv.param.storeEdgePlanes"
      ,"rssv.param.dumpPointsNotPlanes"

      ,"rssv.param.storeBridgesInLocalMemory"
      ,"rssv.param.alignment"
      ,"rssv.param.sfAlignment"      
      ,"rssv.param.sfInterleave"     
      ,"rssv.param.morePlanes"     

      ,"rssv.param.performTraverseSilhouettes"
      ,"rssv.param.exactSilhouetteAABB"
      ,"rssv.param.exactSilhouetteAABBLevel"
      ,"rssv.param.computeSilhouetteBridges"
      ,"rssv.param.computeLastLevelSilhouettes"

      ,"rssv.param.performTraverseTriangles"
      ,"rssv.param.exactTriangleAABB"
      ,"rssv.param.exactTriangleAABBLevel"
      ,"rssv.param.computeTriangleBridges"
      ,"rssv.param.computeLastLevelTriangles"

      ,"rssv.param.performMerge"
      ,"rssv.param.orderedSkala"
      ,"rssv.param.mergeInMega"


      ,"rssv.param.computeSilhouettePlanes"

      );
  std::cerr << "createTraverseSilhouettesProgram" << std::endl;

  auto const adj                          =  vars.get<Adjacency> ("adjacency"                              );
  auto const&cfg                          = *vars.get<Config>    ("rssv.method.config"                     );
  auto const noAABB                       =  vars.getBool        ("rssv.param.noAABB"                      );

  //debug
  auto const storeTraverseSilhouettesStat =  vars.getBool        ("rssv.param.storeTraverseSilhouettesStat");
  auto const storeTraverseTrianglesStat   =  vars.getBool        ("rssv.param.storeTraverseTrianglesStat"  );
  auto const storeEdgePlanes              =  vars.getBool        ("rssv.param.storeEdgePlanes"             );
  auto const dumpPointsNotPlanes          =  vars.getBool        ("rssv.param.dumpPointsNotPlanes"         );

  auto const storeBridgesInLocalMemory    =  vars.getBool        ("rssv.param.storeBridgesInLocalMemory"   );
  auto const alignSize                    =  vars.getSizeT       ("rssv.param.alignment"                   );
  auto const sfAlignment                  =  vars.getUint32      ("rssv.param.sfAlignment"                 );
  auto const sfInterleave                 =  vars.getBool        ("rssv.param.sfInterleave"                );
  auto const morePlanes                   =  vars.getBool        ("rssv.param.morePlanes"                  );

  auto const performTraverseSilhouettes   =  vars.getBool        ("rssv.param.performTraverseSilhouettes"  );
  auto const exactSilhouetteAABB          =  vars.getBool        ("rssv.param.exactSilhouetteAABB"         );
  auto const exactSilhouetteAABBLevel     =  vars.getInt32       ("rssv.param.exactSilhouetteAABBLevel"    );
  auto const computeLastLevelSilhouettes  =  vars.getBool        ("rssv.param.computeLastLevelSilhouettes" );
  auto const computeSilhouetteBridges     =  vars.getBool        ("rssv.param.computeSilhouetteBridges"    );

  auto const performTraverseTriangles     =  vars.getBool        ("rssv.param.performTraverseTriangles"    );
  auto const exactTriangleAABB            =  vars.getBool        ("rssv.param.exactTriangleAABB"           );
  auto const exactTriangleAABBLevel       =  vars.getInt32       ("rssv.param.exactTriangleAABBLevel"      );
  auto const computeLastLevelTriangles    =  vars.getBool        ("rssv.param.computeLastLevelTriangles"   );
  auto const computeTriangleBridges       =  vars.getBool        ("rssv.param.computeTriangleBridges"      );

  auto const computeSilhouettePlanes      =  vars.getBool        ("rssv.param.computeSilhouettePlanes"     );

  auto const performMerge                 =  vars.getBool        ("rssv.param.performMerge"                );
  auto const orderedSkala                 =  vars.getBool        ("rssv.param.orderedSkala"                );
  auto const mergeInMega                  =  vars.getBool        ("rssv.param.mergeInMega"                 );

  auto const nofEdges                     =  adj->getNofEdges();
  auto const nofTriangles                 =  adj->getNofTriangles();

  vars.reCreate<ge::gl::Program>("rssv.method.traverseSilhouettesProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        getConfigShader(vars)
        ,Shader::define("NO_AABB"                       ,(int     )noAABB                      )

        //debug dumps
        ,Shader::define("STORE_SILHOUETTE_TRAVERSE_STAT",(int     )storeTraverseSilhouettesStat)
        ,Shader::define("STORE_TRIANGLE_TRAVERSE_STAT"  ,(int     )storeTraverseTrianglesStat  )
        ,Shader::define("STORE_EDGE_PLANES"             ,(int     )storeEdgePlanes             )

        ,Shader::define("DUMP_POINTS_NOT_PLANES"        ,(int     )dumpPointsNotPlanes         )
        ,Shader::define("STORE_BRIDGES_IN_LOCAL_MEMORY" ,(int     )storeBridgesInLocalMemory   )
        ,Shader::define("USE_BRIDGE_POOL"               ,(int     )cfg.useBridgePool           )
        ,Shader::define("ALIGN_SIZE"                    ,(uint32_t)alignSize                   )
        ,Shader::define("NOF_EDGES"                     ,(uint32_t)nofEdges                    )
        ,Shader::define("NOF_TRIANGLES"                 ,(uint32_t)nofTriangles                )
        ,Shader::define("SF_ALIGNMENT"                  ,(uint32_t)sfAlignment                 )
        ,Shader::define("SF_INTERLEAVE"                 ,(int     )sfInterleave                )
        ,Shader::define("MORE_PLANES"                   ,(int     )morePlanes                  )

        ,Shader::define("PERFORM_TRAVERSE_SILHOUETTES"  ,(int     )performTraverseSilhouettes  )
        ,Shader::define("EXACT_SILHOUETTE_AABB"         ,(int     )exactSilhouetteAABB         )
        ,Shader::define("EXACT_SILHOUETTE_AABB_LEVEL"   ,(int     )exactSilhouetteAABBLevel    )
        ,Shader::define("COMPUTE_LAST_LEVEL_SILHOUETTES",(int     )computeLastLevelSilhouettes )
        ,Shader::define("COMPUTE_SILHOUETTE_BRIDGES"    ,(int     )computeSilhouetteBridges    )

        ,Shader::define("PERFORM_TRAVERSE_TRIANGLES"    ,(int     )performTraverseTriangles    )
        ,Shader::define("EXACT_TRIANGLE_AABB"           ,(int     )exactTriangleAABB           )
        ,Shader::define("EXACT_TRIANGLE_AABB_LEVEL"     ,(int     )exactTriangleAABBLevel      )
        ,Shader::define("COMPUTE_LAST_LEVEL_TRIANGLES"  ,(int     )computeLastLevelTriangles   )
        ,Shader::define("COMPUTE_TRIANGLE_BRIDGES"      ,(int     )computeTriangleBridges      )

        ,Shader::define("COMPUTE_SILHOUETTE_PLANES"     ,(int     )computeSilhouettePlanes     )

        ,Shader::define("PERFORM_MERGE"                 ,(int     )performMerge                )
        ,Shader::define("ORDERED_SKALA"                 ,(int     )orderedSkala                )
        ,Shader::define("MERGE_IN_MEGA"                 ,(int     )mergeInMega                 )

        ,rssv::demortonShader
        ,rssv::mortonShader
        ,rssv::depthToZShader
        ,rssv::quantizeZShader
        ,rssv::getAABBShaderFWD
        ,computeSilhouettePlanes?"":rssv::loadEdgeShaderFWD
        ,rssv::getEdgePlanesShader
        ,rssv::collisionShader
        ,rssv::traverseSilhouettesFWD
        ,rssv::traverseTrianglesFWD
        ,rssv::sharedMemoryShader
        ,rssv::mergeShaderFWD
        ,rssv::globalBarrierShaderFWD
        ,rssv::traverseMain
        ,rssv::globalBarrierShader
        ,rssv::mergeShader
        ,rssv::traverseTriangles
        ,rssv::traverseSilhouettes
        ,rssv::getAABBShader
        ,computeSilhouettePlanes?"":rssv::loadEdgeShader
        ));

}

void createTraverseJobCounters(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method");
  //traverseSilhouettes
  //traverseTriangles
  //traverseDone
  //merge
  vars.reCreate<Buffer>("rssv.method.traverseJobCounters",sizeof(uint32_t)*10);
}


void createDebugSilhouetteTraverseBuffers(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeTraverseSilhouettesStat"
      );
  auto storeTraverseSilhouettesStat = vars.getBool("rssv.param.storeTraverseSilhouettesStat");

  if(storeTraverseSilhouettesStat)
    vars.reCreate<Buffer>("rssv.method.debug.traverseSilhouettesBuffer",sizeof(uint32_t)*(1+1024*1024*128));
  else
    vars.erase("rssv.method.debug.traverseSilhouettesBuffer");
}

void createDebugTriangleTraverseBuffers(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeTraverseTrianglesStat"
      );
  auto storeTraverseTrianglesStat = vars.getBool("rssv.param.storeTraverseTrianglesStat");

  if(storeTraverseTrianglesStat)
    vars.reCreate<Buffer>("rssv.method.debug.traverseTrianglesBuffer",sizeof(uint32_t)*(1+1024*1024*128));
  else
    vars.erase("rssv.method.debug.traverseTrianglesBuffer");
}

void createDebugEdgePlanesBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeEdgePlanes"
      ,"adjacency"
      );
  auto adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Buffer>("rssv.method.debug.edgePlanes",sizeof(uint32_t)*(1+16*adj->getNofEdges()));
}



void traverse(vars::Vars&vars){
  FUNCTION_CALLER();
  createTraverseProgram(vars);
  createTraverseJobCounters(vars);

  //debug
  createDebugSilhouetteTraverseBuffers(vars);
  createDebugTriangleTraverseBuffers(vars);
  createDebugEdgePlanesBuffer(vars);


  auto prg        = vars.get<Program>("rssv.method.traverseSilhouettesProgram");

  auto const&view              = *vars.get<glm::mat4>("rssv.method.viewMatrix"      );
  auto const&proj              = *vars.get<glm::mat4>("rssv.method.projectionMatrix");
  auto const&lightPosition     = *vars.get<glm::vec4>("rssv.method.lightPosition"   );
  auto const clipLightPosition = proj*view*lightPosition;

  auto jobCounters                 = vars.get<Buffer >("rssv.method.traverseJobCounters"       );
  auto stencil                     = vars.get<Texture>("rssv.method.stencil"                   );

  auto const exactSilhouetteAABB        =  vars.getBool        ("rssv.param.exactSilhouetteAABB"       );
  auto const performTraverseSilhouettes =  vars.getBool        ("rssv.param.performTraverseSilhouettes");
  auto const computeSilhouetteBridges   =  vars.getBool        ("rssv.param.computeSilhouetteBridges"  );


  auto const performTraverseTriangles   =  vars.getBool        ("rssv.param.performTraverseTriangles"  );
  auto const computeTriangleBridges     =  vars.getBool        ("rssv.param.computeTriangleBridges"    );

  auto const computeSilhouettePlanes    =  vars.getBool        ("rssv.param.computeSilhouettePlanes"   );

  auto depth      = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask = vars.get<Texture>("shadowMask");

  jobCounters->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
  hierarchy->bindBase(GL_SHADER_STORAGE_BUFFER,0);

  jobCounters      ->bindBase(GL_SHADER_STORAGE_BUFFER,2);

  depth     ->bind     (0);
  shadowMask->bindImage(1);
  stencil   ->bindImage(2);

  prg->use();

  if(performTraverseSilhouettes){
    auto bridges = vars.get<Buffer >("rssv.method.bridges"                   );
    prg->bindBuffer("Bridges",bridges);


    if(computeSilhouetteBridges || exactSilhouetteAABB){
      prg->set4fv      ("clipLightPosition",glm::value_ptr(clipLightPosition));
    }

    auto multBuffer = vars.get<Buffer >("rssv.method.multBuffer"                );
    prg->bindBuffer("MultBuffer",multBuffer);

    if(computeSilhouettePlanes){
      auto sil = vars.get<Buffer>("rssv.method.silhouettePlanes");
      prg->bindBuffer("SilhouettePlanes",sil);
    }else{
      auto invTran = glm::transpose(glm::inverse(proj*view));
      prg->setMatrix4fv("invTran"      ,glm::value_ptr(invTran      ));
      prg->set4fv      ("lightPosition",glm::value_ptr(lightPosition));

      auto projView = proj*view;
      prg->setMatrix4fv("projView"      ,glm::value_ptr(projView      ));

      auto edgePlanes = vars.get<Buffer >("rssv.method.edgePlanes"                );
      prg->bindBuffer("EdgePlanes" ,edgePlanes);
    }
    //prg->set1i("selectedEdge",vars.addOrGetInt32("rssv.param.selectedEdge",-1));
    
    auto const storeTraverseSilhouettesStat = vars.getBool("rssv.param.storeTraverseSilhouettesStat");
    if(storeTraverseSilhouettesStat){
      glFinish();
      auto debug = vars.get<Buffer>("rssv.method.debug.traverseSilhouettesBuffer");
      glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
      prg->bindBuffer("Debug",debug);
    }

    auto const storeEdgePlanes = vars.getBool("rssv.param.storeEdgePlanes");
    if(storeEdgePlanes){
      glFinish();
      auto debug = vars.get<Buffer>("rssv.method.debug.edgePlanes");
      glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
      debug->bindBase(GL_SHADER_STORAGE_BUFFER,7);
    }
  }


  if(performTraverseTriangles){
    auto sf = vars.get<Buffer >("rssv.method.shadowFrusta"    );
    sf->bindBase(GL_SHADER_STORAGE_BUFFER,5);

    auto const storeTraverseTrianglesStat = vars.getBool("rssv.param.storeTraverseTrianglesStat");
    if(storeTraverseTrianglesStat){
      glFinish();
      auto debug = vars.get<Buffer>("rssv.method.debug.traverseTrianglesBuffer");
      glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
      prg->bindBuffer("Debug",debug);
    }

    if(computeTriangleBridges){
      prg->set4fv("clipLightPosition",glm::value_ptr(clipLightPosition));
    }
  }







  auto const compute = [&]{
    auto wg = vars.getUint32("rssv.param.persistentWG");
    glDispatchCompute(wg,1,1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
  };


  if(vars.addOrGetBool("rssv.method.perfCounters.traverseSilhouettes")){
    perf::printComputeShaderProf([&](){
      compute();
    });
  }else{
    compute();
  }


  /*
  uint32_t level = 0;
  uint32_t node = 34;
  std::vector<uint32_t>hi;
  hierarchy->getData(hi);
  auto const ptr = hi.data();
  auto const&cfg = *vars.get<Config>("rssv.method.config");
  uint32_t const*nodePool = ptr;
  float    const*aabbPool = (float*)(ptr + cfg.nodeBufferSize / sizeof(uint32_t));
  uint32_t const*aabbPointer = (uint32_t*)(aabbPool + cfg.aabbBufferSize / sizeof(float));
  float aabb[6];
  for(size_t i=0;i<cfg.floatsPerAABB;++i){
    aabb[i] = aabbPool[aabbPointer[1+cfg.nodeLevelOffset[level]+node]*cfg.floatsPerAABB+i];
  }
  for(size_t i=0;i<cfg.floatsPerAABB;++i){
    std::cerr << aabb[i] << std::endl;
  }
  for(size_t i=0;i<cfg.floatsPerAABB;++i){
    std::cerr << ((uint32_t*)aabb)[i] << std::endl;
  }
  exit(1);
  */
  
  


  //std::vector<uint32_t>jd;
  //jobCounters->getData(jd);
  //for(auto const&x:jd)
  //  std::cerr << x << std::endl;
  //exit(1);

}

}
