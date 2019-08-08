#include <Sintorn2/Sintorn2.h>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <startStop.h>
#include <sstream>
#include <algorithm>
#include <BallotShader.h>

Sintorn2::Sintorn2(vars::Vars& vars) : ShadowMethod(vars) {}

Sintorn2::~Sintorn2() {}

void allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize");

  auto wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto windowSize    = *vars.get<glm::uvec2>("windowSize");
  if(wavefrontSize != 64)throw std::runtime_error("Sintorn2::allocateHierarchy - only 64 warp size supported");

  auto const warpBits  = requiredBits(wavefrontSize);
  auto const warpBitsX = divRoundUp(warpBits,2);
  auto const warpBitsY = warpBits - warpBitsX;
  auto const warpX     = 1 << warpBitsX;
  auto const warpY     = 1 << warpBitsY;
  auto const clusterX  = divRoundUp(windowSize.x,warpX);
  auto const clusterY  = divRoundUp(windowSize.y,warpY);
  auto const wBits     = requiredBits(clusterX);
  auto const hBits     = requiredBits(clusterY);
  auto const dBits     = glm::max(wBits,hBits);
  auto const clusterZ  = 1 << dBits;
  auto const allBits   = wBits + hBits + dBits;
  auto const nofLevels = divRoundUp(allBits,warpBits);

  std::vector<uint32_t>offsets;
  std::vector<uint32_t>sizes;

  for(size_t l=0;l<nofLevels;++l)
    sizes.push_back(1<<glm::max((int)(allBits-warpBits*l),0));
  std::reverse(sizes.begin(),sizes.end());
  uint32_t counter = 0;
  for(size_t i=0;i<sizes.size()-1;++i){
    offsets.push_back(counter);
    counter += sizes[i];
  }

  for(auto const&o:sizes)
    std::cerr << "sizes: " << o << std::endl; 

  for(auto const&o:offsets)
    std::cerr << "offset: " << o << std::endl; 

  size_t bytesPerWarp = wavefrontSize / 8;

  vars.reCreateVector<uint32_t      >("sintorn2.sizes"  ) = sizes  ;
  vars.reCreateVector<uint32_t      >("sintorn2.offsets") = offsets;
  vars.reCreate      <uint32_t      >("sintorn2.allBits",allBits);
  vars.reCreate      <ge::gl::Buffer>("sintorn2.hierarchy",bytesPerWarp * counter);
  vars.reCreate      <uint32_t      >("sintorn2.warpBitsX",warpBitsX);
  vars.reCreate      <uint32_t      >("sintorn2.warpBitsY",warpBitsY);
  vars.reCreate      <uint32_t      >("sintorn2.warpBits" ,warpBits );
}

void createProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize");

  auto const wavefrontSize =  vars.getSizeT           ("wavefrontSize"     );
  auto const windowSize    = *vars.get<glm::uvec2>    ("windowSize"        );
  auto const warpBitsX     =  vars.getUint32          ("sintorn2.warpBitsX");
  auto const warpBitsY     =  vars.getUint32          ("sintorn2.warpBitsY");
  auto const warpBits      =  vars.getUint32          ("sintorn2.warpBits" );
  auto const allBits       =  vars.getUint32          ("sintorn2.allBits"  );
  auto const&offsets       =  vars.getVector<uint32_t>("sintorn2.offsets"  );

  std::stringstream ss;
  ss << "#version 450\n";
  ss << ballotSrc;
  ss << "\n#line " << __LINE__ << std::endl;
  ss << "layout(local_size_x=" << wavefrontSize << ")in;\n";

  ss << R".(
  layout(binding=0)        buffer        Hierarchy{uint hierarchy[];};
  layout(binding=1)uniform sampler2DRect depthTexture;
  uniform float near = 0.01f;
  uniform float far  = 1000.f;
  uniform uint  Sy   = 512/8;
  uniform float fovy = 90;

  // converts depth (-1,+1) to Z in view-space
  // far, near 
  float depthToZ(float d){
    return 2*near*far/(d*(far-near)-far-near);
  }

  // infinite far
  float depthToZInf(float d){
    return 2*near / (d - 1);
  }

  uint convertDepth(float depth){
    return uint(log(-depth/near) / log(1+2*tan(fovy)/Sy));
  }

  uint morton(uvec3 cc){
    uint res = 0;
  ).";

  for(size_t b=0;b<allBits;++b){
    auto const axis = "yxz"[b%3];
    auto const mask = (1<<(size_t)(b/3));
    auto const shift = b - (b/3);
    ss << "res |= (cc." << axis << " & "<< mask <<"u) << " << shift << ";\n";
  }

  ss << R".(
    return res;
  }).";

  ss << "\n#line " << __LINE__ << std::endl;

  ss << R".(
  uint getMorton(){
    float depth = texelFetch(depthTexture,ivec2(gl_GlobalInvocationID.xy)).x*2-1;
    uint  depthQ = convertDepth(depth);
  ).";
  ss << "uvec3 clusterCoord = uvec3(uvec2(gl_GlobalInvocationID) >> uvec2(" << warpBitsX << "," << warpBitsY << "), depthQ);\n";
  ss << R".(  return morton(clusterCoord);
  }).";

  auto const ballotResultInUints = wavefrontSize / (sizeof(uint32_t)*8);
  ss << "shared uint mortons[" << wavefrontSize << "];\n";
  ss << "void compute(){\n";
  ss << "  uint morton = getMorton();\n";
  ss << "  mortons[gl_LocalInvocationIndex] = morton;\n";
  ss << "\n#line " << __LINE__ << std::endl;
  if(ballotResultInUints == 1)
    ss << "  uint notDone = 0xffffffff;\n";
  else
    ss << "  uint notDone[" << ballotResultInUints << "];\n";
  if(ballotResultInUints > 1)
    for(size_t i=0;i<ballotResultInUints;++i)
      ss << "notDone[" << i << "] = 0xffffffff;\n";
  ss << "\n#line " << __LINE__ << std::endl;
  if(ballotResultInUints == 1){
    ss << "while(notDone != 0){\n";
    ss << "  uint selectedBit = findLSB(notDone);\n";
    ss << "  uint otherMorton = mortons[selectedBit];\n";
    ss << "  BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(otherMorton == morton));\n";
    ss << "  if(gl_LocalInvocationIndex == 0){\n";
    ss << "    hierarchy["<< offsets.back() <<"+(morton>>" << warpBits <<")] = GET_UINT_FROM_UINT_ARRAY(sameCluster,0);\n";
    ss << "  }\n";
    ss << "  notDone ^= sameCluster;\n";
    ss << "}\n";
  }else{
    ss << "\n#line " << __LINE__ << std::endl;
    for(size_t i=0;i<ballotResultInUints;++i){
      ss << "while(notDone[" << i << "] != 0){\n";
      ss << "  uint selectedBit = findLSB(notDone["<<i<<"]) + " << i*32 << ";\n";
      ss << "  uint otherMorton = mortons[selectedBit];\n";
      ss << "  BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(otherMorton == morton));\n";
      ss << "  if(gl_LocalInvocationIndex == 0){\n";
      for(size_t j=0;j<ballotResultInUints;++j)
        ss << "    hierarchy[("<< offsets.back() <<"+(morton>>" << warpBits <<"))*" << ballotResultInUints <<"+"<< j <<"] = GET_UINT_FROM_UINT_ARRAY(sameCluster," << j <<");\n";
      ss << "  }\n";
      ss << "  notDone[" << i << "] ^= sameCluster[" << i << "];\n";
      ss << "}\n";
    }
  }
  ss << "}\n";
  ss << "\n#line " << __LINE__ << std::endl;
  ss << "\n";
  ss << "void main(){\n";
  ss << "  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&uint("<<(1<<warpBitsX)-1<<"),uint(gl_LocalInvocationIndex)>>uint("<<warpBitsX<<"));\n";
  ss << "  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2("<<(1<<warpBitsX)<<","<<(1<<warpBitsY)<<");\n";
  ss << "  uvec2 coord = wgCoord + loCoord;\n";
  ss << "  if(any(greaterThanEqual(coord,uvec2("<<windowSize.x<<","<<windowSize.y<<"))))return;\n";
  ss << "  compute();\n";
  ss << "}\n";
  
  std::cerr << ss.str() << std::endl;
  vars.reCreate<ge::gl::Program>("sintorn2.hierarchyProgram0",std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,ss.str()));
}

void Sintorn2::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  allocateHierarchy(vars);
  createProgram(vars);

  auto prg = vars.get<ge::gl::Program>("sintorn2.hierarchyProgram0");

  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  auto hierarchy = vars.get<ge::gl::Buffer>("sintorn2.hierarchy");
  prg->bindBuffer("Hierarchy",hierarchy);
  depth->bind(1);
  
  prg->use();
  glDispatchCompute(1,1,1);

  auto width = depth->getWidth(0);
  auto height = depth->getHeight(0);
  //std::cerr << "width: " << width << " height: " << height << std::endl;
  auto nofPix = width * height;
  std::vector<float>data(nofPix);
  //glGetTextureImage(gBuffer->depth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());
  start(vars,"glGetTextureImage");
  glGetTextureImage(depth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());
  stop(vars,"glGetTextureImage");
  float mmin = 10e10;
  float mmax = -10e10;
  for(auto const&p:data){
    mmin = glm::min(mmin,p);
    mmax = glm::max(mmax,p);
  }
  std::cerr << "mmin: " << mmin << " - mmax: " << mmax << std::endl;
}
