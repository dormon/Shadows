#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <BallotShader.h>
#include <Deferred.h>

#include <RSSV/rasterize.h>
#include <RSSV/rasterizeShader.h>
#include <RSSV/configShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/quantizeZShader.h>
#include <RSSV/depthToZShader.h>
#include <RSSV/traverseSilhouettesShader.h>

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
      ,"windowSize"
      ,"rssv.method.nofTriangles"
      ,"rssv.param.sfWGS"
      ,"rssv.param.triangleAlignment"
      ,"rssv.param.sfAlignment"
      ,"rssv.param.bias"
      ,"rssv.param.sfInterleave"
      ,"rssv.param.triangleInterleave"
      ,"rssv.param.minZBits"
      ,"rssv.param.tileX"   
      ,"rssv.param.tileY"   
      ,"rssv.param.morePlanes"
      ,"rssv.param.ffc"
      ,"rssv.param.noAABB"
      ,"rssv.param.storeTraverseSilhouettesStat"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      );

  auto const wavefrontSize                =  vars.getSizeT       ("wavefrontSize"                          );
  auto const nofTriangles                 =  vars.getUint32      ("rssv.method.nofTriangles"               );
  auto const triangleAlignment            =  vars.getUint32      ("rssv.param.triangleAlignment"           );
  auto const sfAlignment                  =  vars.getUint32      ("rssv.param.sfAlignment"                 );
  auto const sfInterleave                 =  vars.getInt32       ("rssv.param.sfInterleave"                );
  auto const windowSize                   = *vars.get<glm::uvec2>("windowSize"                             );
  auto const tileX                        =  vars.getUint32      ("rssv.param.tileX"                       );
  auto const tileY                        =  vars.getUint32      ("rssv.param.tileY"                       );
  auto const minZBits                     =  vars.getUint32      ("rssv.param.minZBits"                    );
  auto const morePlanes                   =  vars.getInt32       ("rssv.param.morePlanes"                  );
  auto const ffc                          =  vars.getInt32       ("rssv.param.ffc"                         );
  auto const noAABB                       =  vars.getInt32       ("rssv.param.noAABB"                      );
  auto const storeTraverseSilhouettesStat =  vars.getBool        ("rssv.param.storeTraverseSilhouettesStat");
  auto const nnear                        =  vars.getFloat       ("args.camera.near"                       );
  auto const ffar                         =  vars.getFloat       ("args.camera.far"                        );
  auto const fovy                         =  vars.getFloat       ("args.camera.fovy"                       );

  vars.reCreate<ge::gl::Program>("rssv.method.traverseSilhouettesProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WARP"               ,(uint32_t)wavefrontSize     ),
        Shader::define("NOF_TRIANGLES"      ,(uint32_t)nofTriangles      ),
        Shader::define("TRIANGLE_ALIGNMENT" ,(uint32_t)triangleAlignment ),
        Shader::define("SF_ALIGNMENT"       ,(uint32_t)sfAlignment       ),
        Shader::define("SF_INTERLEAVE"      ,(int)     sfInterleave      ),
        Shader::define("WINDOW_X"           ,(uint32_t)windowSize.x      ),
        Shader::define("WINDOW_Y"           ,(uint32_t)windowSize.y      ),
        Shader::define("MIN_Z_BITS"         ,(uint32_t)minZBits          ),
        Shader::define("TILE_X"             ,tileX                       ),
        Shader::define("TILE_Y"             ,tileY                       ),
        Shader::define("MORE_PLANES"        ,(int)     morePlanes        ),
        Shader::define("ENABLE_FFC"         ,(int)     ffc               ),
        Shader::define("NO_AABB"            ,(int)     noAABB            ),
#if SAVE_COLLISION == 1
        Shader::define("SAVE_COLLISION"     ,(int)1),
#endif
        Shader::define("STORE_TRAVERSE_STAT",(int)storeTraverseSilhouettesStat),
        Shader::define("NEAR"      ,nnear                  ),
        glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
        Shader::define("FOVY"      ,fovy                   ),
        ballotSrc,
        rssv::configShader,
        rssv::demortonShader,
        rssv::depthToZShader,
        rssv::quantizeZShader,
        rssv::traverseSilhouettesShader
        ));

}

void createSilhouetteJobCounter(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method");
  vars.reCreate<Buffer>("rssv.method.silhouettesJobCounter",sizeof(uint32_t));
}


void createDebugSilhouetteTraverseBuffers(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.param.storeTraverseSilhouettesStat"
      );
  vars.reCreate<Buffer>("rssv.method.debug.traverseSilhouettesBuffer",sizeof(uint32_t)*(1+1024*1024*128));
}



void traverseSilhouettes(vars::Vars&vars){
  FUNCTION_CALLER();
  createTraverseSilhouettesProgram(vars);
  createSilhouetteJobCounter(vars);
  createDebugSilhouetteTraverseBuffers(vars);

  auto prg        = vars.get<Program>("rssv.method.traverseSilhouettesProgram");
  auto nodePool   = vars.get<Buffer >("rssv.method.nodePool"                  );
  auto aabbPool   = vars.get<Buffer >("rssv.method.aabbPool"                  );
  auto sf         = vars.get<Buffer >("rssv.method.shadowFrusta"              );
  auto jobCounter = vars.get<Buffer >("rssv.method.silhouettesJobCounter"     );
  auto depth      = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask = vars.get<Texture>("shadowMask");

  jobCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  nodePool  ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool  ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  sf        ->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  jobCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  depth     ->bind(0);
  shadowMask->bindImage(1);

  float data[1] = {1.f};
  vars.get<ge::gl::Texture>("shadowMask")->clear(0,GL_RED,GL_FLOAT,data);

  prg->use();

  auto const storeTraverseStat = vars.getBool("rssv.param.storeTraverseSilhouettesStat");
  if(storeTraverseStat){
    glFinish();
    auto debug = vars.get<Buffer>("rssv.method.debug.traverseSilhouettesBuffer");
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
