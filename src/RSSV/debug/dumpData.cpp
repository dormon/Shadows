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
#include <RSSV/debug/dumpSamples.h>
#include <RSSV/buildHierarchy.h>
#include <RSSV/traverseSilhouettes.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void dumpNodePool(vars::Vars&vars){
  FUNCTION_CALLER();
  auto const cfg           = *vars.get<Config>("rssv.method.debug.dump.config");
  auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.nodePool",cfg.nodeBufferSize);
  ge::gl::glCopyNamedBufferSubData(hierarchy->getId(),buf->getId(),cfg.nodeBufferOffsetInHierarchy,0,buf->getSize());
}

void dumpAABBPointer(vars::Vars&vars){
  FUNCTION_CALLER();
  auto const cfg           = *vars.get<Config>("rssv.method.debug.dump.config");
  if(!cfg.memoryOptim)return;

  auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.aabbPointer",cfg.aabbPointerBufferSize);
  ge::gl::glCopyNamedBufferSubData(hierarchy->getId(),buf->getId(),cfg.aabbPointerBufferOffsetInHierarchy,0,buf->getSize());
}

void dumpBridges(vars::Vars&vars){
  FUNCTION_CALLER();
  auto toBackup  = vars.get<Buffer >("rssv.method.bridges"              );
  auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.bridges",toBackup->getSize());
  buf->copy(*toBackup);
  //std::vector<int>data;
  //buf->getData(data);
  //for(auto const&x:data)
  //  if(x!=0)std::cerr << x << std::endl;
  //exit(0);
}

void dumpAABBPool(vars::Vars&vars){
  FUNCTION_CALLER();
  auto const cfg           = *vars.get<Config>("rssv.method.debug.dump.config");

    auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
    auto buf = vars.reCreate<Buffer>("rssv.method.debug.dump.aabbPool",cfg.aabbBufferSize);
    ge::gl::glCopyNamedBufferSubData(hierarchy->getId(),buf->getId(),cfg.aabbBufferOffsetInHierarchy,0,buf->getSize());
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

void dumpBasic(vars::Vars&vars){
  FUNCTION_CALLER();
  auto lp        = *vars.get<glm::vec4>("rssv.method.debug.lightPosition"   );
  auto vm        = *vars.get<glm::mat4>("rssv.method.debug.viewMatrix"      );
  auto pm        = *vars.get<glm::mat4>("rssv.method.debug.projectionMatrix");

  auto cfg       = *vars.get<Config>("rssv.method.config");


  vars.reCreate<glm::vec4 >("rssv.method.debug.dump.lightPosition"   ,lp           );
  vars.reCreate<glm::mat4 >("rssv.method.debug.dump.viewMatrix"      ,vm           );
  vars.reCreate<glm::mat4 >("rssv.method.debug.dump.projectionMatrix",pm           );
  vars.reCreate<Config    >("rssv.method.debug.dump.config"          ,cfg          ); 
}

void dumpTraversePlanes(vars::Vars&vars){
  vars.getBool("rssv.param.storeEdgePlanes") = true;
  vars.updateTicks("rssv.param.storeEdgePlanes");
  traverseSilhouettes(vars);
  vars.getBool("rssv.param.storeEdgePlanes") = false;
  vars.updateTicks("rssv.param.storeEdgePlanes");

  auto debug = vars.get<Buffer>("rssv.method.debug.edgePlanes");
  uint32_t NN = 0;
  debug->getData(&NN,sizeof(uint32_t));
  std::vector<float>debugData((NN*16+1)*sizeof(float));
  debug->getData(debugData.data(),(NN*16+1)*sizeof(float));

  std::cerr << "N: " << NN << std::endl;
  for(size_t i=0;i<NN*16;i+=16){
    for(size_t j=0;j<16;j+=4){
      char const*name[] = {
        "ed",
        "ap",
        "bp",
        "ab",
      };
      std::cerr << name[j/4] << ": ";
      for(size_t k=0;k<4;k+=1)
        std::cerr << debugData[1+i+j+k] << " ";
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
  }

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
}

}
