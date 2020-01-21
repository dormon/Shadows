#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
#include <RSSV/collisionShader.h>
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
      ,"rssv.param.sfWGS"
      ,"rssv.param.bias"
      ,"rssv.param.triangleInterleave"
      ,"rssv.param.minZBits"
      ,"rssv.param.tileX"   
      ,"rssv.param.tileY"   
      ,"rssv.param.noAABB"
      ,"rssv.param.storeTraverseSilhouettesStat"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      ,"rssv.method.alignedNofEdges"
      );

  auto const wavefrontSize                =  vars.getSizeT       ("wavefrontSize"                          );
  auto const windowSize                   = *vars.get<glm::uvec2>("windowSize"                             );
  auto const tileX                        =  vars.getUint32      ("rssv.param.tileX"                       );
  auto const tileY                        =  vars.getUint32      ("rssv.param.tileY"                       );
  auto const minZBits                     =  vars.getUint32      ("rssv.param.minZBits"                    );
  auto const noAABB                       =  vars.getInt32       ("rssv.param.noAABB"                      );
  auto const storeTraverseSilhouettesStat =  vars.getBool        ("rssv.param.storeTraverseSilhouettesStat");
  auto const nnear                        =  vars.getFloat       ("args.camera.near"                       );
  auto const ffar                         =  vars.getFloat       ("args.camera.far"                        );
  auto const fovy                         =  vars.getFloat       ("args.camera.fovy"                       );
  auto const alignedNofEdges              =  vars.getUint32      ("rssv.method.alignedNofEdges"            );

  vars.reCreate<ge::gl::Program>("rssv.method.traverseSilhouettesProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WARP"               ,(uint32_t)wavefrontSize     ),
        Shader::define("WINDOW_X"           ,(uint32_t)windowSize.x      ),
        Shader::define("WINDOW_Y"           ,(uint32_t)windowSize.y      ),
        Shader::define("MIN_Z_BITS"         ,(uint32_t)minZBits          ),
        Shader::define("TILE_X"             ,tileX                       ),
        Shader::define("TILE_Y"             ,tileY                       ),
        Shader::define("NO_AABB"            ,(int)     noAABB            ),
        Shader::define("STORE_TRAVERSE_STAT",(int)storeTraverseSilhouettesStat),
        Shader::define("NEAR"      ,nnear                  ),
        glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
        Shader::define("FOVY"      ,fovy                   ),
        Shader::define("ALIGNED_NOF_EDGES",alignedNofEdges),
        ballotSrc,
        rssv::configShader,
        rssv::demortonShader,
        rssv::depthToZShader,
        rssv::quantizeZShader,
        rssv::collisionShader,
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

  auto const view              = *vars.get<glm::mat4>("rssv.method.viewMatrix"      );
  auto const proj              = *vars.get<glm::mat4>("rssv.method.projectionMatrix");
  auto const lightPosition     = *vars.get<glm::vec4>("rssv.method.lightPosition"   );

  auto nodePool          = vars.get<Buffer >("rssv.method.nodePool"             );
  auto aabbPool          = vars.get<Buffer >("rssv.method.aabbPool"             );
  auto jobCounter        = vars.get<Buffer >("rssv.method.silhouettesJobCounter");
  auto edges             = vars.get<Buffer >("rssv.method.edgeBuffer"           );
  auto multBuffer        = vars.get<Buffer >("rssv.method.multBuffer"           );
  auto silhouetteCounter = vars.get<Buffer >("rssv.method.silhouetteCounter"    );
  auto bridges           = vars.get<Buffer >("rssv.method.bridges"              );
  auto stencil           = vars.get<Texture>("rssv.method.stencil"              );

  auto depth      = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask = vars.get<Texture>("shadowMask");

  jobCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  bridges->clear(GL_R32I,GL_RED_INTEGER,GL_INT);
  stencil->clear(0,GL_RED_INTEGER,GL_INT);

  nodePool         ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool         ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  jobCounter       ->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  edges            ->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  multBuffer       ->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  silhouetteCounter->bindBase(GL_SHADER_STORAGE_BUFFER,5);
  bridges          ->bindBase(GL_SHADER_STORAGE_BUFFER,6);

  //std::vector<uint32_t>sil;
  //multBuffer->getData(sil);
  //for(auto const&x:sil){
  //  auto res = x;
  //  uint32_t edge = res & 0x1fffffffu;                                         
  //  int  mult = int(res) >> 29;  
  //  std::cerr << edge << " - " << mult << std::endl;
  //}
  //exit(1);

  depth     ->bind     (0);
  shadowMask->bindImage(1);
  stencil   ->bindImage(2);

  float data[1] = {1.f};
  vars.get<ge::gl::Texture>("shadowMask")->clear(0,GL_RED,GL_FLOAT,data);

  prg->use();

  //auto invTran = glm::transpose(glm::inverse(proj*view));
  prg
    ->setMatrix4fv("view"         ,glm::value_ptr(view         ))
    ->setMatrix4fv("proj"         ,glm::value_ptr(proj         ))
    //->setMatrix4fv("invTran"      ,glm::value_ptr(invTran      ))
    ->set4fv      ("lightPosition",glm::value_ptr(lightPosition));

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