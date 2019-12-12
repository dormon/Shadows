#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <BallotShader.h>
#include <Deferred.h>

#include <Sintorn2/rasterize.h>
#include <Sintorn2/rasterizeShader.h>
#include <Sintorn2/configShader.h>
#include <Sintorn2/mortonShader.h>

#include <iomanip>

using namespace ge::gl;
using namespace std;

namespace sintorn2{
void createRasterizeProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"wavefrontSize"
      ,"sintorn2.method.nofTriangles"
      ,"sintorn2.param.sfWGS"
      ,"sintorn2.param.triangleAlignment"
      ,"sintorn2.param.sfAlignment"
      ,"sintorn2.param.bias"
      ,"sintorn2.param.sfInterleave"
      ,"sintorn2.param.triangleInterleave"
      );

  auto const wavefrontSize       = vars.getSizeT ("wavefrontSize"                    );
  auto const nofTriangles        = vars.getUint32("sintorn2.method.nofTriangles"     );
  auto const triangleAlignment   = vars.getUint32("sintorn2.param.triangleAlignment" );
  auto const sfAlignment         = vars.getUint32("sintorn2.param.sfAlignment"       );
  auto const sfInterleave        = vars.getInt32 ("sintorn2.param.sfInterleave"      );

  vars.reCreate<ge::gl::Program>("sintorn2.method.rasterizeProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WARP"               ,(uint32_t)wavefrontSize     ),
        Shader::define("NOF_TRIANGLES"      ,(uint32_t)nofTriangles      ),
        Shader::define("TRIANGLE_ALIGNMENT" ,(uint32_t)triangleAlignment ),
        Shader::define("SF_ALIGNMENT"       ,(uint32_t)sfAlignment       ),
        Shader::define("SF_INTERLEAVE"      ,(int)     sfInterleave      ),
        ballotSrc,
        sintorn2::demortonShader,
        sintorn2::configShader,
        sintorn2::rasterizeShader
        ));

}

void createJobCounter(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method");
  vars.reCreate<Buffer>("sintorn2.method.jobCounter",sizeof(uint32_t));
}

}


void sintorn2::rasterize(vars::Vars&vars){
  FUNCTION_CALLER();
  createRasterizeProgram(vars);
  createJobCounter(vars);

  auto prg        = vars.get<Program>("sintorn2.method.rasterizeProgram");
  auto nodePool   = vars.get<Buffer >("sintorn2.method.nodePool"        );
  auto aabbPool   = vars.get<Buffer >("sintorn2.method.aabbPool"        );
  auto sf         = vars.get<Buffer >("sintorn2.method.shadowFrusta"    );
  auto jobCounter = vars.get<Buffer >("sintorn2.method.jobCounter"      );
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

  auto deb = make_shared<Buffer>(sizeof(float)*7*10000);
  auto debc = make_shared<Buffer>(sizeof(uint32_t));
  debc->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  deb->bindBase(GL_SHADER_STORAGE_BUFFER,6);
  debc->bindBase(GL_SHADER_STORAGE_BUFFER,7);

#if 1
  glDispatchCompute(1024,1,1);
  glMemoryBarrier(GL_ALL_BARRIER_BITS);
#else

  glDispatchCompute(1,1,1);
  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  //*
  std::vector<float>debData;
  std::vector<uint32_t>debcData;
  deb->getData(debData);
  debc->getData(debcData);
  for(size_t i=0;i<debcData[0];++i){
    uint32_t const N=17;
    float*dd = debData.data()+i*N;
#define FF std::setprecision(4) << std::fixed << std::showpos
    std::cerr << "aabb: " << FF << dd[0] << " "<< dd[1] << " "<< dd[2] << " - ";
    std::cerr << FF << dd[3] << " "<< dd[4] << " "<< dd[5] << " # ";
    std::cerr << "plane: " << FF << dd[6] << " "<< dd[7] << " "<< dd[8] << " "<< dd[9] << " ";
    std::cerr << "tr: " << FF << dd[10] << " "<< dd[11] << " "<< dd[12] << " ";
    std::cerr << "ta: " << FF << dd[13] << " "<< dd[14] << " "<< dd[15] << " ";

    auto plane = glm::vec4(dd[6],dd[7],dd[8],dd[9]);
    auto tr = glm::vec4(dd[10],dd[11],dd[12],1.f);
    auto ta = glm::vec4(dd[13],dd[14],dd[15],1.f);

    std::cerr << "dot: " << glm::dot(plane,tr) <<  " - " << glm::dot(plane,ta) << " ";
    std::cerr << "thread: " << dd[16] << " ";
    std::cerr << std::endl;
  }

  // */

  exit(1);
#endif
}
