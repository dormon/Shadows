#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <imguiDormon/imgui.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>

#include <Sintorn2/dumpData.h>

using namespace ge::gl;
using namespace std;

namespace sintorn2::debug{

void createCopyViewSamplesProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug");
  std::string const cs = 
  R".(
  #version 450
  
  layout(local_size_x=16,local_size_y=16)in;
  
  layout(binding=0)uniform usampler2D colorTexture;
  layout(binding=1)uniform  sampler2D positionTexture;
  layout(binding=2)uniform  sampler2D normalTexture;
  
  layout(binding=0)buffer Samples{float samples[];};
  
  uniform mat4 view = mat4(1);
  uniform mat4 proj = mat4(1);

  uniform uvec2 windowSize = uvec2(512,512);
  
  void main(){
    if(any(greaterThanEqual(uvec2(gl_GlobalInvocationID.xy),windowSize)))return;

    uint sampleId = gl_GlobalInvocationID.y * windowSize.x + gl_GlobalInvocationID.x;

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    vec3 position = texelFetch(positionTexture,coord,0).xyz;
    vec3 normal   = texelFetch(normalTexture  ,coord,0).xyz;
    uvec4 color   = texelFetch(colorTexture   ,coord,0);
    vec3  Ka      = vec3((color.xyz>>0u)&0xffu)/0xffu;

    samples[sampleId*9+0+0] = position[0];
    samples[sampleId*9+0+1] = position[1];
    samples[sampleId*9+0+2] = position[2];
    samples[sampleId*9+3+0] = normal  [0];
    samples[sampleId*9+3+1] = normal  [1];
    samples[sampleId*9+3+2] = normal  [2];
    samples[sampleId*9+6+0] = Ka      [0];
    samples[sampleId*9+6+1] = Ka      [1];
    samples[sampleId*9+6+2] = Ka      [2];

  }
  ).";


  vars.reCreate<Program>("sintorn2.method.debug.copyViewSamplesData",
      make_shared<Shader>(GL_COMPUTE_SHADER,cs));

}

void createViewSamplesBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");

  auto const nofSamples = windowSize.x*windowSize.y;
  auto const floatsPerSample = 3 + 3 + 3;
  auto const bufSize = nofSamples * floatsPerSample * sizeof(float);
  vars.reCreate<Buffer>("sintorn2.method.debug.samples",bufSize);
}

void dumpSamples(vars::Vars&vars){
  createCopyViewSamplesProgram(vars);
  createViewSamplesBuffer(vars);

  auto prg = vars.get<Program>("sintorn2.method.debug.copyViewSamplesData");
  auto buf = vars.get<Buffer >("sintorn2.method.debug.samples");
  auto gBuffer = vars.get<GBuffer>("gBuffer");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");

  buf->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  gBuffer->color   ->bind(0);
  gBuffer->position->bind(1);
  gBuffer->normal  ->bind(2);
  prg->set2ui("windowSize",windowSize.x,windowSize.y);
  prg->use();

  glDispatchCompute(divRoundUp(windowSize.x,16),divRoundUp(windowSize.y,16),1);
  glFinish();

}

void dumpData(vars::Vars&vars){
  dumpSamples(vars);
  std::cerr << "dump" << std::endl;
}

}
