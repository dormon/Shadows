#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <imguiDormon/imgui.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>

#include <RSSV/config.h>
#include <RSSV/debug/dumpData.h>
#include <RSSV/buildHierarchy.h>
#include <RSSV/traverseSilhouettes.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void createCopyViewSamplesProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug");
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


  vars.reCreate<Program>("rssv.method.debug.copyViewSamplesData",
      make_shared<Shader>(GL_COMPUTE_SHADER,cs));

}

void createViewSamplesBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");

  auto const nofSamples = windowSize.x*windowSize.y;
  auto const floatsPerSample = 3 + 3 + 3;
  auto const bufSize = nofSamples * floatsPerSample * sizeof(float);
  vars.reCreate<Buffer>("rssv.method.debug.dump.samples",bufSize);
}

void dumpSamples(vars::Vars&vars){
  FUNCTION_CALLER();
  createCopyViewSamplesProgram(vars);
  createViewSamplesBuffer(vars);

  auto prg = vars.get<Program>("rssv.method.debug.copyViewSamplesData");
  auto buf = vars.get<Buffer >("rssv.method.debug.dump.samples");
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

void dumpNodePool(vars::Vars&vars){
  FUNCTION_CALLER();
  auto nodePool = vars.get<Buffer>("rssv.method.nodePool");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.nodePool",nodePool->getSize());
  buf->copy(*nodePool);
}

void dumpAABBPool(vars::Vars&vars){
  FUNCTION_CALLER();
  auto aabbPool = vars.get<Buffer>("rssv.method.aabbPool");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.aabbPool",aabbPool->getSize());
  buf->copy(*aabbPool);
}

void dumpSF(vars::Vars&vars){
  FUNCTION_CALLER();
  auto sf = vars.get<Buffer>("rssv.method.shadowFrusta");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.shadowFrusta",sf->getSize());
  buf->copy(*sf);
}

void dumpEdges(vars::Vars&vars){
  FUNCTION_CALLER();
  auto edges = vars.get<Buffer>("rssv.method.edgeBuffer");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.edgeBuffer",edges->getSize());
  buf->copy(*edges);
}

void dumpMultBuffer(vars::Vars&vars){
  FUNCTION_CALLER();
  auto source = vars.get<Buffer>("rssv.method.multBuffer");
  auto target = vars.reCreate<Buffer>("rssv.method.debug.dump.multBuffer",source->getSize());
  target->copy(*source);
}

void dumpSilhouetteCounter(vars::Vars&vars){
  FUNCTION_CALLER();
  auto source = vars.get<Buffer>("rssv.method.silhouetteCounter");
  auto target = vars.reCreate<Buffer>("rssv.method.debug.dump.silhouetteCounter",source->getSize());
  target->copy(*source);
}

void dumpBasic(vars::Vars&vars){
  FUNCTION_CALLER();
  auto lp        = *vars.get<glm::vec4>("rssv.method.debug.lightPosition"   );
  auto vm        = *vars.get<glm::mat4>("rssv.method.debug.viewMatrix"      );
  auto pm        = *vars.get<glm::mat4>("rssv.method.debug.projectionMatrix");
  auto const nnear =  vars.getFloat("args.camera.near");
  auto const ffar  =  vars.getFloat("args.camera.far" );
  auto const fovy  =  vars.getFloat("args.camera.fovy");

  auto cfg       = *vars.get<Config>("rssv.method.config");


  vars.reCreate<glm::vec4 >("rssv.method.debug.dump.lightPosition"   ,lp   );
  vars.reCreate<glm::mat4 >("rssv.method.debug.dump.viewMatrix"      ,vm   );
  vars.reCreate<glm::mat4 >("rssv.method.debug.dump.projectionMatrix",pm   );
  vars.reCreate<float     >("rssv.method.debug.dump.near"            ,nnear);
  vars.reCreate<float     >("rssv.method.debug.dump.far"             ,ffar );
  vars.reCreate<float     >("rssv.method.debug.dump.fovy"            ,fovy );
  vars.reCreate<Config    >("rssv.method.debug.dump.config"          ,cfg  ); 
}

void dumpTraverse(vars::Vars&vars){
  vars.getBool("rssv.param.storeTraverseSilhouettesStat") = true;
  vars.updateTicks("rssv.param.storeTraverseSilhouettesStat");
  traverseSilhouettes(vars);
  vars.getBool("rssv.param.storeTraverseSilhouettesStat") = false;
  vars.updateTicks("rssv.param.storeTraverseSilhouettesStat");


  auto debug = vars.get<Buffer>("rssv.method.debug.traverseSilhouettesBuffer");
  uint32_t NN = 0;
  debug->getData(&NN,sizeof(uint32_t));
  std::vector<uint32_t>debugData((NN*4+1)*sizeof(uint32_t));
  debug->getData(debugData.data(),(NN*4+1)*sizeof(uint32_t));

  //PRINT FIRST 100 INTERSECTIONS
  //std::cerr << "NONO:" << debugData[0] << std::endl;
  //for(size_t i=0;i<debugData[0];++i)
  //  std::cerr << debugData[1+i*4+0] << " "<< debugData[1+i*4+1] << " "<< debugData[1+i*4+2] << " "<< debugData[1+i*4+3] << " " << std::endl;

  std::map<uint32_t,std::vector<uint32_t>>taData;
  std::map<uint32_t,std::vector<uint32_t>>trData;
  std::map<uint32_t,std::vector<uint32_t>>inData;

  std::cerr << "N: " << debugData[0] << std::endl;
  std::map<uint32_t,uint32_t>levelCnt;
  std::map<uint32_t,uint32_t>jobCnt;
  std::map<uint32_t,uint32_t>nodeCnt;
  std::map<uint32_t,uint32_t>statCnt;
  std::map<uint32_t,std::map<uint32_t,uint32_t>>levelStatCnt;
  uint32_t const taStat = 3;
  uint32_t const trStat = 0xf0;
  uint32_t const inStat = 2;
  uint32_t const saStat = 0xff;
  uint32_t const emStat = 0;
  for(uint32_t j=0;j<debugData[0];++j){
    auto i = 1+j*4;
    auto job   = debugData[i+0];
    auto node  = debugData[i+1];
    auto level = debugData[i+2];
    auto stat  = debugData[i+3];
    
    //if(job == 1){
      if(stat == taStat){
        if(taData.count(level) == 0)taData[level] = std::vector<uint32_t>();
        taData[level].push_back(node);
      }

      if(stat == trStat){
        if(trData.count(level) == 0)trData[level] = std::vector<uint32_t>();
        trData[level].push_back(node);
      }

      if(stat == inStat){
        if(inData.count(level) == 0)inData[level] = std::vector<uint32_t>();
        inData[level].push_back(node);
      }
    //}


    if(levelCnt.count(level)==0)levelCnt[level] = 0;
    if(jobCnt  .count(job  )==0)jobCnt  [job  ] = 0;
    if(nodeCnt .count(node )==0)nodeCnt [node ] = 0;
    if(statCnt .count(stat )==0)statCnt [stat ] = 0;
    if(levelStatCnt.count(level)==0)levelStatCnt[level] = std::map<uint32_t,uint32_t>();
    if(levelStatCnt[level].count(stat) == 0)levelStatCnt[level][stat] = 0;
    levelCnt[level]++;
    jobCnt  [job  ]++;
    nodeCnt [node ]++;
    statCnt [stat ]++;
    levelStatCnt[level][stat]++;
  }
  for(auto const&x:levelCnt){
    std::cerr << "level" << x.first << ": " << x.second << std::endl;
  }

  //for(auto const&x:jobCnt){
  //  std::cerr << "job" << x.first << ": " << x.second << std::endl;
  //}

  auto statToName = [&](uint32_t s){
    if(s == emStat)return "empty  ";
    if(s == taStat)return "ta     ";
    if(s == inStat)return "in     ";
    if(s == trStat)return "re     ";
    if(s == saStat)return "samples";
    return              "bug    ";
  };
  //for(auto const&x:nodeCnt){
  //  std::cerr << "node" << x.first << ": " << x.second << std::endl;
  //}
  for(auto const&x:statCnt){
    std::cerr << "stat " << statToName(x.first) << ": " << x.second << std::endl;
  }
  for(auto const&l:levelStatCnt){
    std::cerr << "level" << l.first << ": " << std::endl;
    for(auto const&s:l.second)
      std::cerr << "  " << statToName(s.first) << ": " << s.second << std::endl;
  }
  //exit(0);

  //for(uint32_t i=0;i<10000;++i)
  //  std::cerr << debugData[i] << " ";
  //std::cerr << std::endl;
  //auto cfg = *vars.get<Config>("rssv.method.config");;
  //for(uint32_t j=0;j<debugData[0];++j){
  //  auto i = 1+j*3;
  //  auto job   = debugData[i+0];
  //  auto node  = debugData[i+1];
  //  auto level = debugData[i+2];
  //  if(level > cfg.nofLevels){
  //    std::cerr << "job  : " << job   << " ";
  //    std::cerr << "node : " << node  << " ";
  //    std::cerr << "level: " << level << std::endl;
  //  }
  //}
 

  auto const cfg = *vars.get<Config>("rssv.method.debug.dump.config"    );

  //|ta_dip0|tr_dip0|in_dip0|ta_dip1|tr_dip1|in_dip1|ta_dip2|tr_dip2|in_dip2|

  std::vector<uint32_t>traverseData;
  uint32_t offset = 0;
  uint32_t const dipOffset = (cfg.nofLevels*4)*3;

  auto const pushEmpty = [&](){
    traverseData.push_back(0);
    traverseData.push_back(0);
    traverseData.push_back(0);
    traverseData.push_back(0);
  };
  auto const pushDIP = [&](uint32_t n){
    traverseData.push_back(1);
    traverseData.push_back(n);
    traverseData.push_back(offset+dipOffset);
    traverseData.push_back(0);
    offset += n;
  };

  auto const pushOne = [&](uint32_t l,std::map<uint32_t,std::vector<uint32_t>>const&w){
    if(w.count(l))pushDIP(w.at(l).size());
    else          pushEmpty();
  };

  for(uint32_t l=0;l<cfg.nofLevels;++l){
    pushOne(l,taData);
    pushOne(l,trData);
    pushOne(l,inData);
  }

  auto const pushNodes = [&](uint32_t l,std::map<uint32_t,std::vector<uint32_t>>const&w){
    if(w.count(l))
      for(uint32_t i=0;i<w.at(l).size();++i)
        traverseData.push_back(w.at(l).at(i));
  };

  for(uint32_t l=0;l<cfg.nofLevels;++l){
    pushNodes(l,taData);
    pushNodes(l,trData);
    pushNodes(l,inData);
  }

  //for(uint32_t i=0;i<4*3;++i)
  //  for(uint32_t k=0;k<4;++k){
  //    char const* names[] = {
  //      "vertices",
  //      "instances",
  //      "firstVertex",
  //      "baseInstance",
  //    };
  //    std::cerr << names[k] << ": " << traverseData[i*4+k] << std::endl;
  //  }

  vars.reCreate<Buffer>("rssv.method.debug.traverseData",traverseData);

}

void dumpData(vars::Vars&vars){
  FUNCTION_CALLER();
  dumpSamples(vars);
  dumpBasic(vars);

  //dumpSF(vars);


  buildHierarchy(vars);
  dumpNodePool(vars);
  dumpAABBPool(vars);

  dumpTraverse(vars);


  std::cerr << "dump" << std::endl;
}

void dumpSilhouettes(vars::Vars&vars){
  //dumpEdges(vars);
  dumpMultBuffer(vars);
  dumpSilhouetteCounter(vars);
}

}
