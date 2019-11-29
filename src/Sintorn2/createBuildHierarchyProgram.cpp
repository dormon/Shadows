#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <Sintorn2/createBuildHierarchyProgram.h>

void sintorn2::createBuildHierarchyProgram(vars::Vars&vars){
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
    ss << "  activeThread = uint(all(lessThan(coord,uvec2("<<windowSize.x<<","<<windowSize.y<<"))));\n";
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
  ss << "uint activeThread = 0;\n";

  ss << convertDepth();
  ss << morton();

  DEBUG_SHADER_LINE();

  ss << getMorton();


  ss << "shared uint sharedMorton;\n";
  ss << "\n";

  ss << "void compute(uvec2 coord){\n";
  ss << "  uint morton = getMorton(coord);\n";
  //ss << "  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;\n";
  //ss << "  float z = depthToZInf(depth);\n";
  //ss << "  uint  zQ = quantizeZ(z);\n";
  //ss << "  hierarchy[(gl_WorkGroupID.x + gl_WorkGroupID.y*64)*64+gl_LocalInvocationIndex] = floatBitsToUint(z);\n";
  //ss << "  hierarchy[(gl_WorkGroupID.x + gl_WorkGroupID.y*64)*64+gl_LocalInvocationIndex] = zQ;\n";
  //ss << "  return;\n";
  //ss << "  mortons[gl_LocalInvocationIndex] = morton;\n";

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
    ss << "  notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),0);\n";
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
      ss << "  notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),"<<i<<");\n";
      ss << "  while(notDone != 0){\n";
      ss << "    if(gl_LocalInvocationIndex == findLSB(notDone) + " << i*32 << ")\n";
      ss << "      sharedMorton = morton;\n";
      ss << "    uint otherMorton = sharedMorton;\n";
      ss << "    BALLOT_UINTS sameCluster = BALLOT_RESULT_TO_UINTS(BALLOT(otherMorton == morton && activeThread != 0));\n";
      ss << "    if(gl_LocalInvocationIndex == 0){\n";
      for(size_t j=0;j<warpInUints;++j){
        ss << "      hierarchy["<< offsets.back() << "+otherMorton*" << warpInUints <<"+"<< j <<"] = GET_UINT_FROM_UINT_ARRAY(sameCluster," << j <<");\n";
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
