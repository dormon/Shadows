#include <Sintorn/ShadowFrusta.h>
#include <Barrier.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Model.h>
#include <util.h>
#include <FunctionPrologue.h>

using namespace std;
using namespace ge::gl;

size_t const VEC4_PER_SHADOWFRUSTUM   = 6;
size_t const FLOATS_PER_SHADOWFRUSTUM = VEC4_PER_SHADOWFRUSTUM*4;

void allocateShadowFrustaBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","model");

  vector<float>vertices;
  vars.get<Model>("model")->getVertices(vertices);
  auto nofTriangles = vertices.size()/3/3;
  
  vars.reCreate<size_t>("sintorn.nofTriangles",nofTriangles);
  vars.reCreate<Buffer>("sintorn.shadowFrusta",sizeof(float)*FLOATS_PER_SHADOWFRUSTUM*nofTriangles);
  auto triangles = vars.reCreate<Buffer>("sintorn.triangles",sizeof(float)*4*3*nofTriangles);


  //allocate triangles
  float*Ptr=(float*)triangles->map();
  for(unsigned t=0;t<nofTriangles;++t)
    for(unsigned p=0;p<3;++p){
      for(unsigned k=0;k<3;++k)
        Ptr[(t*3+p)*4+k]=vertices[(t*3+p)*3+k];
      Ptr[(t*3+p)*4+3]=1;
    }
  triangles->unmap();
}

#include<Sintorn/ShadowFrustaShaders.h>

void createShadowFrustaProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","args.sintorn.bias","args.sintorn.shadowFrustaWGS");

  vars.reCreate<Program>("sintorn.sfProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("BIAS"          ,float   (vars.getFloat ("args.sintorn.bias")           )),
        Shader::define("WAVEFRONT_SIZE",uint32_t(vars.getUint32("args.sintorn.shadowFrustaWGS"))),
        sintorn::shadowFrustaShader));
}

void computeShadowFrusta(vars::Vars&vars,glm::vec4 const&lightPosition,glm::mat4 mvp){
  FUNCTION_CALLER();

  allocateShadowFrustaBuffer(vars);
  createShadowFrustaProgram(vars);
  vars.get<Buffer>("sintorn.triangles")->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  vars.get<Buffer>("sintorn.shadowFrusta")->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  vars.get<Program>("sintorn.sfProgram")
    ->set1ui      ("nofTriangles"                       ,static_cast<uint32_t>(vars.getSizeT("sintorn.nofTriangles"))   )
    ->setMatrix4fv("modelViewProjection"                ,glm::value_ptr(mvp)                                            )
    ->set4fv      ("lightPosition"                      ,glm::value_ptr(lightPosition)                                  )
    ->setMatrix4fv("transposeInverseModelViewProjection",glm::value_ptr(glm::inverse(glm::transpose(mvp)))              )
    //->bindBuffer  ("triangles"                          ,vars.get<Buffer>("sintorn.triangles")                          )
    //->bindBuffer  ("shadowFrusta"                       ,vars.get<Buffer>("sintorn.shadowFrusta")                       )
    ->dispatch    (getDispatchSize(vars.getSizeT("sintorn.nofTriangles"),vars.getUint32("args.sintorn.shadowFrustaWGS")));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
