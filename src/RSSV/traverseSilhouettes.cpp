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
#include <RSSV/getAABBShader.h>
#include <RSSV/loadEdgeShader.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>
#include <RSSV/config.h>
#include <perfCounters.h>

using namespace ge::gl;
using namespace std;

namespace rssv{

void createTraverseSilhouettesProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"rssv.param.sfWGS"
      ,"rssv.param.bias"
      ,"rssv.param.triangleInterleave"
      ,"rssv.param.noAABB"
      ,"rssv.param.storeTraverseSilhouettesStat"
      ,"rssv.param.storeEdgePlanes"
      ,"rssv.method.config"
      ,"rssv.param.dumpPointsNotPlanes"
      ,"rssv.param.computeBridges"
      ,"rssv.param.storeBridgesInLocalMemory"
      ,"rssv.param.computeLastLevelSilhouettes"
      ,"rssv.param.alignment"
      ,"adjacency"
      );

  auto const noAABB                       =  vars.getInt32       ("rssv.param.noAABB"                      );
  auto const storeTraverseSilhouettesStat =  vars.getBool        ("rssv.param.storeTraverseSilhouettesStat");
  auto const storeEdgePlanes              =  vars.getBool        ("rssv.param.storeEdgePlanes"             );
  auto const dumpPointsNotPlanes          =  vars.getBool        ("rssv.param.dumpPointsNotPlanes"         );
  auto const computeBridges               =  vars.getBool        ("rssv.param.computeBridges"              );
  auto const storeBridgesInLocalMemory    =  vars.getBool        ("rssv.param.storeBridgesInLocalMemory"   );
  auto const computeLastLevelSilhouettes  =  vars.getBool        ("rssv.param.computeLastLevelSilhouettes" );
  auto const&cfg                          = *vars.get<Config>    ("rssv.method.config"                     );
  auto const alignSize                    =  vars.getSizeT       ("rssv.param.alignment"                   );
  auto adj                                =  vars.get<Adjacency> ("adjacency"                              );
  auto const nofEdges                     =  adj->getNofEdges();
  auto const nofTriangles                 =  adj->getNofTriangles();


  vars.reCreate<ge::gl::Program>("rssv.method.traverseSilhouettesProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        getConfigShader(vars),
        Shader::define("NO_AABB"                       ,(int)     noAABB                 ),
        Shader::define("STORE_TRAVERSE_STAT"           ,(int)storeTraverseSilhouettesStat),
        Shader::define("STORE_EDGE_PLANES"             ,(int)storeEdgePlanes             ),
        Shader::define("DUMP_POINTS_NOT_PLANES"        ,(int)dumpPointsNotPlanes         ),
        Shader::define("COMPUTE_BRIDGES"               ,(int)computeBridges              ),
        Shader::define("STORE_BRIDGES_IN_LOCAL_MEMORY" ,(int)storeBridgesInLocalMemory   ),
        Shader::define("USE_BRIDGE_POOL"               ,(int)cfg.useBridgePool           ),
        Shader::define("COMPUTE_LAST_LEVEL_SILHOUETTES",(int)computeLastLevelSilhouettes ),
        Shader::define("ALIGN_SIZE"                    ,(uint32_t)alignSize              ),
        Shader::define("NOF_EDGES"                     ,(uint32_t)nofEdges               ),
        Shader::define("NOF_TRIANGLES"                 ,(uint32_t)nofTriangles           )
        ,rssv::demortonShader
        ,rssv::depthToZShader
        ,rssv::quantizeZShader
        ,rssv::collisionShader
        ,rssv::getAABBShaderFWD
        ,rssv::loadEdgeShaderFWD
        ,rssv::getEdgePlanesShader
        ,rssv::traverseSilhouettesShader
        ,rssv::getAABBShader
        ,rssv::loadEdgeShader
        ));

}

void createTriangleBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method","adjacency");
  auto const adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Buffer>("rssv.method.triangles",adj->getVertices());
}

void createTraverseJobCounters(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method");
  vars.reCreate<Buffer>("rssv.method.traverseJobCounters",sizeof(uint32_t)*2);
}


void createDebugSilhouetteTraverseBuffers(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeTraverseSilhouettesStat"
      );
  vars.reCreate<Buffer>("rssv.method.debug.traverseSilhouettesBuffer",sizeof(uint32_t)*(1+1024*1024*128));
}

void createDebugEdgePlanesBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeEdgePlanes"
      ,"adjacency"
      );
  auto adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Buffer>("rssv.method.debug.edgePlanes",sizeof(uint32_t)*(1+16*adj->getNofEdges()));
}



void traverseSilhouettes(vars::Vars&vars){
  FUNCTION_CALLER();
  createTraverseSilhouettesProgram(vars);
  createTraverseJobCounters(vars);
  createDebugSilhouetteTraverseBuffers(vars);
  createDebugEdgePlanesBuffer(vars);
  createTriangleBuffer(vars);

  auto prg        = vars.get<Program>("rssv.method.traverseSilhouettesProgram");

  auto const&view              = *vars.get<glm::mat4>("rssv.method.viewMatrix"      );
  auto const&proj              = *vars.get<glm::mat4>("rssv.method.projectionMatrix");
  auto const&lightPosition     = *vars.get<glm::vec4>("rssv.method.lightPosition"   );
  auto const clipLightPosition = proj*view*lightPosition;

  auto jobCounters                 = vars.get<Buffer >("rssv.method.traverseJobCounters"       );
  auto edgePlanes                  = vars.get<Buffer >("rssv.method.edgePlanes"                );
  auto multBuffer                  = vars.get<Buffer >("rssv.method.multBuffer"                );
  auto bridges                     = vars.get<Buffer >("rssv.method.bridges"                   );
  auto triangles                   = vars.get<Buffer >("rssv.method.triangles"                 );
  auto stencil                     = vars.get<Texture>("rssv.method.stencil"                   );
  auto computeBridges              = vars.getBool     ("rssv.param.computeBridges"             );


  auto depth      = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask = vars.get<Texture>("shadowMask");

  jobCounters->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  bridges->clear(GL_R32I,GL_RED_INTEGER,GL_INT);
  stencil->clear(0,GL_RED_INTEGER,GL_INT);

  auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
  hierarchy->bindBase(GL_SHADER_STORAGE_BUFFER,0);

  jobCounters      ->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  edgePlanes       ->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  multBuffer       ->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  bridges          ->bindBase(GL_SHADER_STORAGE_BUFFER,6);

  depth     ->bind     (0);
  shadowMask->bindImage(1);
  stencil   ->bindImage(2);

  float data[1] = {1.f};
  vars.get<ge::gl::Texture>("shadowMask")->clear(0,GL_RED,GL_FLOAT,data);

  prg->use();

  auto invTran = glm::transpose(glm::inverse(proj*view));

  prg->setMatrix4fv("invTran"      ,glm::value_ptr(invTran      ));
  prg->set4fv      ("lightPosition",glm::value_ptr(lightPosition));

  if(computeBridges){
    auto projView = proj*view;
    prg->setMatrix4fv("projView"      ,glm::value_ptr(projView      ));
    prg->set4fv      ("clipLightPosition",glm::value_ptr(clipLightPosition));
  }

  prg->bindBuffer("Triangles",triangles);

  //prg->set1i("selectedEdge",vars.addOrGetInt32("rssv.param.selectedEdge",-1));

  auto const storeTraverseStat = vars.getBool("rssv.param.storeTraverseSilhouettesStat");
  if(storeTraverseStat){
    glFinish();
    auto debug = vars.get<Buffer>("rssv.method.debug.traverseSilhouettesBuffer");
    glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    debug->bindBase(GL_SHADER_STORAGE_BUFFER,7);
  }

  auto const storeEdgePlanes = vars.getBool("rssv.param.storeEdgePlanes");
  if(storeEdgePlanes){
    glFinish();
    auto debug = vars.get<Buffer>("rssv.method.debug.edgePlanes");
    glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    debug->bindBase(GL_SHADER_STORAGE_BUFFER,7);
  }





  if(vars.addOrGetBool("rssv.method.perfCounters.traverseSilhouettes")){
    perf::printComputeShaderProf([&](){
      glDispatchCompute(1024,1,1);
      glMemoryBarrier(GL_ALL_BARRIER_BITS);
    });
  }else{
    glDispatchCompute(1024,1,1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
  }

}

}
