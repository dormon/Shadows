#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <Model.h>
#include <divRoundUp.h>
#include <align.h>
#include <perfCounters.h>

#include <RSSV/computeShadowFrusta.h>
#include <RSSV/shadowFrustaShader.h>

using namespace ge::gl;
using namespace std;

namespace rssv{
void createShadowFrustaProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"rssv.method.nofTriangles"
      ,"rssv.param.sfWGS"
      ,"rssv.param.triangleAlignment"
      ,"rssv.param.sfAlignment"
      ,"rssv.param.bias"
      ,"rssv.param.sfInterleave"
      ,"rssv.param.triangleInterleave"
      ,"rssv.param.morePlanes"
      ,"rssv.param.ffc"
      );

  auto const wavefrontSize       = vars.getSizeT ("wavefrontSize"                    );
  auto const nofTriangles        = vars.getUint32("rssv.method.nofTriangles"     );
  auto const sfWGS               = vars.getUint32("rssv.param.sfWGS"             );
  auto const triangleAlignment   = vars.getUint32("rssv.param.triangleAlignment" );
  auto const sfAlignment         = vars.getUint32("rssv.param.sfAlignment"       );
  auto const bias                = vars.getFloat ("rssv.param.bias"              );
  auto const sfInterleave        = vars.getInt32 ("rssv.param.sfInterleave"      );
  auto const triangleInterleave  = vars.getInt32 ("rssv.param.triangleInterleave");
  auto const morePlanes          = vars.getInt32 ("rssv.param.morePlanes"        );
  auto const ffc                 = vars.getInt32 ("rssv.param.ffc"               );


  vars.reCreate<ge::gl::Program>("rssv.method.shadowFrustaProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WARP"               ,(uint32_t)wavefrontSize     ),
        Shader::define("NOF_TRIANGLES"      ,(uint32_t)nofTriangles      ),
        Shader::define("WGS"                ,(uint32_t)sfWGS             ),
        Shader::define("TRIANGLE_ALIGNMENT" ,(uint32_t)triangleAlignment ),
        Shader::define("SF_ALIGNMENT"       ,(uint32_t)sfAlignment       ),
        Shader::define("BIAS"               ,(float   )bias              ),
        Shader::define("SF_INTERLEAVE"      ,(int)     sfInterleave      ),
        Shader::define("TRIANGLE_INTERLEAVE",(int)     triangleInterleave),
        Shader::define("MORE_PLANES"        ,(int)     morePlanes        ),
        Shader::define("ENABLE_FFC"         ,(int)     ffc               ),
        rssv::shadowFrustaShader
        ));
}

void allocateShadowFrusta(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"rssv.param.triangleAlignment"
      ,"rssv.param.sfAlignment"
      ,"rssv.param.triangleInterleave"
      ,"rssv.param.morePlanes"
      ,"rssv.param.ffc"
      ,"model"
      );

  auto const triangleAlignment   = vars.getUint32("rssv.param.triangleAlignment");
  auto const sfAlignment         = vars.getUint32("rssv.param.sfAlignment"      );
  auto const triangleInterleave  = vars.getInt32 ("rssv.param.triangleInterleave");
  auto const morePlanes          = vars.getInt32 ("rssv.param.morePlanes"        );
  auto const ffc                 = vars.getInt32 ("rssv.param.ffc"               );

  vector<float>vertices = vars.get<Model>("model")->getVertices();
  auto nofTriangles = (uint32_t)(vertices.size()/3/3);
  
  uint32_t const planesPerSF = 4 + morePlanes*3;
  uint32_t const floatsPerPlane = 4;
  uint32_t const floatsPerSF = floatsPerPlane * planesPerSF + (uint32_t)ffc;

  auto const aNofT = align(nofTriangles,(uint32_t)triangleAlignment);

  std::vector<float>triData(aNofT * 3 * 3);

  if(triangleInterleave == 1){
    for(uint32_t p=0;p<3;++p)
      for(uint32_t k=0;k<3;++k)
        for(uint32_t t=0;t<nofTriangles;++t)triData[aNofT*(p*3+k)+t] = vertices[(t*3+p)*3+k];
  }else{
    for(uint32_t t=0;t<nofTriangles;++t)
      for(uint32_t p=0;p<3;++p)
        for(uint32_t k=0;k<3;++k)
          triData[(t*3+p)*3+k] = vertices[(t*3+p)*3+k];
  }

  auto const aNofSF = align(nofTriangles,(uint32_t)sfAlignment);
  uint32_t const sfSize = sizeof(float)*floatsPerSF*aNofSF;

  vars.reCreate<Buffer  >("rssv.method.shadowFrusta",sfSize      );
  vars.reCreate<Buffer  >("rssv.method.triangles"   ,triData     );
  vars.reCreate<uint32_t>("rssv.method.nofTriangles",nofTriangles);
}

}

void rssv::computeShadowFrusta(vars::Vars&vars){
  allocateShadowFrusta(vars);
  createShadowFrustaProgram(vars);
  //exit(1);

  auto triangles           =  vars.get<Buffer>   ("rssv.method.triangles"          );
  auto sf                  =  vars.get<Buffer>   ("rssv.method.shadowFrusta"       );
  auto const nofTriangles  =  vars.getUint32     ("rssv.method.nofTriangles"       );
  auto const lightPosition = *vars.get<glm::vec4>("rssv.method.lightPosition"      );
  auto const viewMatrix    = *vars.get<glm::mat4>("rssv.method.viewMatrix"         );
  auto const projMatrix    = *vars.get<glm::mat4>("rssv.method.projectionMatrix"   );
  auto const sfWGS         =  vars.getUint32     ("rssv.param.sfWGS"               );
  auto       prg           =  vars.get<Program>  ("rssv.method.shadowFrustaProgram");

  auto const mvp = projMatrix * viewMatrix;

  triangles->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  sf       ->bindBase(GL_SHADER_STORAGE_BUFFER,1);


  prg
    ->set4fv      ("lightPosition"                      ,glm::value_ptr(lightPosition)                    )
    ->setMatrix4fv("transposeInverseModelViewProjection",glm::value_ptr(glm::inverse(glm::transpose(mvp))))
    ->use();

  if(vars.addOrGetBool("rssv.method.perfCounters.shadowFrusta")){
    if(vars.addOrGetBool("rssv.method.perfCounters.oneCounter")){
      perf::printComputeShaderProf([&](){
      glDispatchCompute(divRoundUp(nofTriangles,sfWGS),1,1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      },vars.addOrGetUint32("rssv.method.perfCounters.counter"));
    }else{
      perf::printComputeShaderProf([&](){
      glDispatchCompute(divRoundUp(nofTriangles,sfWGS),1,1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      });
    }
  }else{
    glDispatchCompute(divRoundUp(nofTriangles,sfWGS),1,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }

}
