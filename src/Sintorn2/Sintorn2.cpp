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
  auto const xBits     = requiredBits(clusterX);
  auto const yBits     = requiredBits(clusterY);
  auto const zBits     = glm::max(glm::max((uint32_t)xBits,(uint32_t)yBits),9u);
  auto const clusterZ  = 1 << zBits;
  auto const allBits   = xBits + yBits + zBits;
  auto const nofLevels = divRoundUp(allBits,warpBits);
  auto const warpInUints = wavefrontSize / (sizeof(uint32_t)*8);


  std::vector<uint32_t>counts ;
  for(size_t l=0;l<=nofLevels;++l)
    counts.push_back(1<<glm::max((int)(allBits-warpBits*l),0));
  std::reverse(counts.begin(),counts.end());


  std::vector<uint32_t>sizes  ;
  for(auto const&x:counts)
    sizes.push_back(x*warpInUints);


  std::vector<uint32_t>offsets;
  uint32_t counter = 0;
  for(size_t i=0;i<sizes.size();++i){
    offsets.push_back(counter);
    counter += sizes[i];
  }


  for(auto const&o:counts)
    std::cerr << "counts: " << o << std::endl; 

  for(auto const&o:sizes)
    std::cerr << "sizes: " << o << std::endl; 

  for(auto const&o:offsets)
    std::cerr << "offset: " << o << std::endl; 

  vars.reCreateVector<uint32_t      >("sintorn2.counts" ) = counts  ;
  vars.reCreateVector<uint32_t      >("sintorn2.offsets") = offsets ;
  vars.reCreateVector<uint32_t      >("sintorn2.sizes"  ) = sizes   ;
  vars.reCreate      <uint32_t      >("sintorn2.allBits",allBits);
  vars.reCreate      <uint32_t      >("sintorn2.warpBitsX",warpBitsX);
  vars.reCreate      <uint32_t      >("sintorn2.warpBitsY",warpBitsY);
  vars.reCreate      <uint32_t      >("sintorn2.warpBits" ,warpBits );
  vars.reCreate      <uint32_t      >("sintorn2.xBits"    ,xBits    );
  vars.reCreate      <uint32_t      >("sintorn2.yBits"    ,yBits    );
  vars.reCreate      <uint32_t      >("sintorn2.zBits"    ,zBits    );
  vars.reCreate      <uint32_t      >("sintorn2.clusterX",clusterX);
  vars.reCreate      <uint32_t      >("sintorn2.clusterY",clusterY);
  vars.reCreate      <ge::gl::Buffer>("sintorn2.hierarchy",counter * warpInUints * sizeof(uint32_t));
}

void createProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize");

  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"     );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"        );
  auto const warpBitsX           =  vars.getUint32          ("sintorn2.warpBitsX");
  auto const warpBitsY           =  vars.getUint32          ("sintorn2.warpBitsY");
  auto const warpBits            =  vars.getUint32          ("sintorn2.warpBits" );
  auto const allBits             =  vars.getUint32          ("sintorn2.allBits"  );
  auto const xBits               =  vars.getUint32          ("sintorn2.xBits"    );
  auto const yBits               =  vars.getUint32          ("sintorn2.yBits"    );
  auto const zBits               =  vars.getUint32          ("sintorn2.zBits"    );
  auto const&offsets             =  vars.getVector<uint32_t>("sintorn2.offsets"  );
  auto const warpInUints         = wavefrontSize / (sizeof(uint32_t)*8);

  auto morton = [&](){
    std::stringstream ss;
    
    ss << "uint morton(uvec3 cc){\n";
    ss << "  uint res = 0;\n";

    size_t usedX = 0;
    size_t usedY = 0;
    size_t usedZ = 0;
    size_t used[] = {0,0,0};
    size_t bits[] = {xBits,yBits,zBits};
    size_t axis = 0;
    for(size_t b=0;b<allBits;++b){
      while(used[axis] == bits[axis])axis = (axis+1)%3;
      auto const axisName = "xyz"[axis];
      auto const mask = (1<<used[axis]);
      auto const shift = b - used[axis];
      ss << "  res |= (cc." << axisName << " & "<< mask <<"u) << " << shift << ";\n";
      used[axis]++;
      axis = (axis+1)%3;
    }

    ss << "  return res;\n";
    ss << "}\n";
    ss << "\n";

    return ss.str();
  };

  auto convertDepth = [&](){
    std::stringstream ss;

    ss << "uniform float near = 0.01f;\n";
    ss << "uniform float far  = 1000.f;\n";
    ss << "uniform uint  Sy   = 512/8;\n";
    ss << "uniform float fovy = 1.5707963267948966f;\n";
    ss << "uniform uint  zCluster = 1<<6;\n";
    ss << "\n";
    ss << "// converts depth (-1,+1) to Z in view-space\n";
    ss << "// far, near \n";
    ss << "float depthToZ(float d){\n";
    ss << "  return 2*near*far/(d*(far-near)-far-near);\n";
    ss << "}\n";
    ss << "\n";
    ss << "// infinite far\n";
    ss << "float depthToZInf(float d){\n";
    ss << "  return 2*near / (d - 1);\n";
    ss << "}\n";
    ss << "\n";
    ss << "uint quantizeZ(float z){\n";
    ss << "  return clamp(uint(log(-z/near) / log(1+2*tan(fovy/2)/Sy)),0,zCluster-1);\n";
    ss << "}\n";
    ss << "\n";

    return ss.str();
  };

  auto getMorton = [&](){
    std::stringstream ss;

    ss << "uint getMorton(uvec2 coord){\n";
    ss << "  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;\n";
    ss << "  float z = depthToZInf(depth);\n";
    ss << "  uint  zQ = quantizeZ(z);\n";
    ss << "  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(" << warpBitsX << "," << warpBitsY << "), zQ);\n";
    ss << "  return morton(clusterCoord);\n";
    ss << "}\n";
    ss << "\n";

    return ss.str();
  };

  auto writeMain = [&](){
    std::stringstream ss;

    ss << "void main(){\n";
    ss << "  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&"<<(1<<warpBitsX)-1<<"u,uint(gl_LocalInvocationIndex)>>"<<warpBitsX<<"u);\n";
    ss << "  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2("<<(1<<warpBitsX)<<","<<(1<<warpBitsY)<<");\n";
    ss << "  uvec2 coord = wgCoord + loCoord;\n";
    ss << "  if(any(greaterThanEqual(coord,uvec2("<<windowSize.x<<","<<windowSize.y<<"))))return;\n";
    ss << "  compute(coord);\n";
    ss << "}\n";
    ss << "\n";

    return ss.str();
  };

#define DEBUG_SHADER_LINE() ss << "\n#line " << __LINE__ << std::endl

  std::stringstream ss;
  ss << "#version 450\n";

  ss << ballotSrc;

  DEBUG_SHADER_LINE();

  ss << "layout(local_size_x=" << wavefrontSize << ")in;\n";

  ss << "layout(binding=0)        buffer        Hierarchy{uint hierarchy[];};\n";
  ss << "layout(binding=1)uniform sampler2DRect depthTexture;\n";

  ss << convertDepth();
  ss << morton();

  DEBUG_SHADER_LINE();

  ss << getMorton();


  ss << "shared uint mortons[" << wavefrontSize << "];\n";
  ss << "\n";

  ss << "void compute(uvec2 coord){\n";
  ss << "  uint morton = getMorton(coord);\n";
  //ss << "  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;\n";
  //ss << "  float z = depthToZInf(depth);\n";
  //ss << "  uint  zQ = quantizeZ(z);\n";
  //ss << "  hierarchy[(gl_WorkGroupID.x + gl_WorkGroupID.y*64)*64+gl_LocalInvocationIndex] = floatBitsToUint(z);\n";
  //ss << "  hierarchy[(gl_WorkGroupID.x + gl_WorkGroupID.y*64)*64+gl_LocalInvocationIndex] = zQ;\n";
  //ss << "  return;\n";
  ss << "  mortons[gl_LocalInvocationIndex] = morton;\n";

  DEBUG_SHADER_LINE();

  //ss << "  uint notDone = 0xffffffff;\n";
  ss << "  uint notDone;\n";

  DEBUG_SHADER_LINE();
  //
  //   1x1
  //   8x8
  //  64x64
  // 512x512 = 64x64 * 8x8
  // allBits = 6+6+6 = 18
  // [zyxzyx][zyxzyx][zyxzyx]
  // hierarchy[(0+1+64+64*64)+(morton>> 0)]  = sameCluster
  // hierarchy[(0+1+64      )+(morton>> 6)] |= 1 << (morton>> 0)&(64-1)
  // hierarchy[(0+1         )+(morton>>12)] |= 1 << (morton>> 6)&(64-1)
  // hierarchy[(0           )+(morton>>18)] |= 1 << (morton>>12)&(64-1)
  // 
  //
  if(warpInUints == 1){
    ss << "  notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(true)),0);\n";
    ss << "  while(notDone != 0){\n";
    ss << "    uint selectedBit = findLSB(notDone);\n";
    ss << "    uint otherMorton = mortons[selectedBit];\n";
    ss << "    BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(otherMorton == morton));\n";
    ss << "    if(gl_LocalInvocationIndex == 0){\n";
    ss << "      hierarchy["<< offsets.back() <<"+otherMorton] = GET_UINT_FROM_UINT_ARRAY(sameCluster,0);\n";
    for(size_t i=0;i<offsets.size()-1;++i){
      auto const offset = offsets[offsets.size()-2-i];
      ss << "      atomicOr(hierarchy["<< offset<<"+(otherMorton>>("<<warpBits*(i+1)<<"))],(otherMorton>>"<<warpBits*i<<")&"<<((1<<warpBits)-1)<<");\n";
    }
    ss << "    }\n";
    ss << "    notDone ^= sameCluster;\n";
    ss << "  }\n";
  }else{
    DEBUG_SHADER_LINE();
    ss << "uint counter = 0;\n";
    for(size_t i=0;i<warpInUints;++i){
      ss << "  notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(true)),"<<i<<");\n";
      ss << "  while(notDone != 0){\n";
      ss << "    uint selectedBit = findLSB(notDone) + " << i*32 << ";\n";
      ss << "    uint otherMorton = mortons[selectedBit];\n";
      ss << "    BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(otherMorton == morton));\n";
      ss << "    if(gl_LocalInvocationIndex == 0){\n";
      for(size_t j=0;j<warpInUints;++j){
        ss << "      hierarchy["<< offsets.back() << "+otherMorton*" << warpInUints <<"+"<< j <<"] = GET_UINT_FROM_UINT_ARRAY(sameCluster," << j <<");\n";
        //ss << "      hierarchy["<< offsets.back() << "+otherMorton*" << warpInUints <<"+"<< j <<"] = 0xffffffff;\n";
      }
      ss << "      uint bit;\n";
      for(size_t i=0;i<offsets.size()-1;++i){
        auto const offset = offsets[offsets.size()-2-i];
        if(warpBits*(i+1) >= allBits)
          ss << "      atomicOr(hierarchy[otherMorton>>5],1<<(otherMorton&31u));\n";
        else{
          ss << "      bit = otherMorton&"<<((1<<warpBits)-1)<<"u;\n";
          ss << "      otherMorton >>= " << warpBits << "u;\n";
          ss << "      atomicOr(hierarchy["<< offset<<"+otherMorton*"<<warpInUints<<"+(bit>>5)],1<<(bit&31u));\n";
        }
      }
      ss << "    }\n";
      ss << "    notDone ^= sameCluster[" << i << "];\n";
      ss << "  }\n";
    }
  }
  ss << "}\n";
  DEBUG_SHADER_LINE();
  ss << "\n";

  ss << writeMain();
  
  std::cerr << ss.str() << std::endl;
  //exit(0);
  vars.reCreate<ge::gl::Program>("sintorn2.hierarchyProgram0",std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,ss.str()));
#undef DEBUG_SHADER_LINE
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
  hierarchy->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  prg->bindBuffer("Hierarchy",hierarchy);
  depth->bind(1);
  

  prg->use();
  auto const clusterX = vars.getUint32("sintorn2.clusterX");
  auto const clusterY = vars.getUint32("sintorn2.clusterY");
  glDispatchCompute(clusterX,clusterY,1);
  std::vector<uint32_t>ddd;
  hierarchy->getData(ddd);
  std::cerr << ddd[0] << std::endl;
  std::cerr << ddd[1] << std::endl;

  /*
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

  // */
}
