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
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>
#include <Sintorn2/config.h>
#include <perfCounters.h>

using namespace ge::gl;
using namespace std;

//#define SAVE_COLLISION 1

#include <string.h>

namespace sintorn2{
void createRasterizeProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"wavefrontSize"
      ,"windowSize"
      ,"sintorn2.method.nofTriangles"
      ,"sintorn2.param.sfWGS"
      ,"sintorn2.param.triangleAlignment"
      ,"sintorn2.param.sfAlignment"
      ,"sintorn2.param.bias"
      ,"sintorn2.param.sfInterleave"
      ,"sintorn2.param.triangleInterleave"
      ,"sintorn2.param.minZBits"
      ,"sintorn2.param.tileX"   
      ,"sintorn2.param.tileY"   
      ,"sintorn2.param.morePlanes"
      ,"sintorn2.param.ffc"
      ,"sintorn2.param.noAABB"
      ,"sintorn2.param.storeTraverseStat"
      ,"sintorn2.param.memoryOptim"
      ,"sintorn2.param.taOptim"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      );

  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"                    );
  auto const nofTriangles        =  vars.getUint32          ("sintorn2.method.nofTriangles"     );
  auto const triangleAlignment   =  vars.getUint32          ("sintorn2.param.triangleAlignment" );
  auto const sfAlignment         =  vars.getUint32          ("sintorn2.param.sfAlignment"       );
  auto const sfInterleave        =  vars.getInt32           ("sintorn2.param.sfInterleave"      );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"                       );
  auto const tileX               =  vars.getUint32          ("sintorn2.param.tileX"             );
  auto const tileY               =  vars.getUint32          ("sintorn2.param.tileY"             );
  auto const minZBits            =  vars.getUint32          ("sintorn2.param.minZBits"          );
  auto const morePlanes          =  vars.getInt32           ("sintorn2.param.morePlanes"        );
  auto const ffc                 =  vars.getInt32           ("sintorn2.param.ffc"               );
  auto const noAABB              =  vars.getInt32           ("sintorn2.param.noAABB"            );
  auto const storeTraverseStat   =  vars.getBool            ("sintorn2.param.storeTraverseStat" );
  auto const memoryOptim         =  vars.getInt32           ("sintorn2.param.memoryOptim"       );
  auto const taOptim             =  vars.getInt32           ("sintorn2.param.taOptim"           );
  auto const nnear               =  vars.getFloat           ("args.camera.near"                 );
  auto const ffar                =  vars.getFloat           ("args.camera.far"                  );
  auto const fovy                =  vars.getFloat           ("args.camera.fovy"                 );

  vars.reCreate<ge::gl::Program>("sintorn2.method.rasterizeProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WARP"               ,(uint32_t)wavefrontSize     ),
        Shader::define("NOF_TRIANGLES"      ,(uint32_t)nofTriangles      ),
        Shader::define("TRIANGLE_ALIGNMENT" ,(uint32_t)triangleAlignment ),
        Shader::define("SF_ALIGNMENT"       ,(uint32_t)sfAlignment       ),
        Shader::define("SF_INTERLEAVE"      ,(int)     sfInterleave      ),
        Shader::define("WINDOW_X"           ,(uint32_t)windowSize.x      ),
        Shader::define("WINDOW_Y"           ,(uint32_t)windowSize.y      ),
        Shader::define("MIN_Z_BITS"         ,(uint32_t)minZBits          ),
        Shader::define("TILE_X"             ,tileX                       ),
        Shader::define("TILE_Y"             ,tileY                       ),
        Shader::define("MORE_PLANES"        ,(int)     morePlanes        ),
        Shader::define("ENABLE_FFC"         ,(int)     ffc               ),
        Shader::define("NO_AABB"            ,(int)     noAABB            ),
        Shader::define("USE_TA_OPTIM"       ,(int)     taOptim           ),

#if SAVE_COLLISION == 1
        Shader::define("SAVE_COLLISION"     ,(int)1),
#endif
        Shader::define("STORE_TRAVERSE_STAT",(int)storeTraverseStat),
        Shader::define("MEMORY_OPTIM"       ,(int)memoryOptim      ),

        Shader::define("NEAR"      ,nnear                  ),
        glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
        Shader::define("FOVY"      ,fovy                   ),
        ballotSrc,
        sintorn2::configShader,
        sintorn2::demortonShader,
        sintorn2::depthToZShader,
        sintorn2::quantizeZShader,
        sintorn2::rasterizeShader
        ));

}

#if SAVE_COLLISION == 1
std::shared_ptr<Buffer>deb;
std::shared_ptr<Buffer>debc;
const uint32_t floatsPerStore = 1 + 1 + 3 + 3 + (4+3)*4;
#endif

void createJobCounter(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method");
  vars.reCreate<Buffer>("sintorn2.method.jobCounter",sizeof(uint32_t));

#if SAVE_COLLISION == 1
  deb = make_shared<Buffer>(sizeof(float)*floatsPerStore*10000);
  debc = make_shared<Buffer>(sizeof(uint32_t)*(1));
#endif
}

}

void createDebugTraverseBuffers(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"sintorn2.param.storeTraverseStat"
      );
  vars.reCreate<Buffer>("sintorn2.method.debug.traverseBuffer",sizeof(uint32_t)*(1+1024*1024*128));
}


void sintorn2::rasterize(vars::Vars&vars){
  FUNCTION_CALLER();
  createRasterizeProgram(vars);
  createJobCounter(vars);
  createDebugTraverseBuffers(vars);

  auto prg         = vars.get<Program>("sintorn2.method.rasterizeProgram");
  auto nodePool    = vars.get<Buffer >("sintorn2.method.nodePool"        );
  auto aabbPool    = vars.get<Buffer >("sintorn2.method.aabbPool"        );
  auto sf          = vars.get<Buffer >("sintorn2.method.shadowFrusta"    );
  auto jobCounter  = vars.get<Buffer >("sintorn2.method.jobCounter"      );
  auto depth       = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask  = vars.get<Texture>("shadowMask");
  auto memoryOptim = vars.getInt32    ("sintorn2.param.memoryOptim"      );


  jobCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  nodePool  ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool  ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  sf        ->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  jobCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  depth     ->bind(0);
  shadowMask->bindImage(1);

  if(memoryOptim){
    auto aabbPointer = vars.get<Buffer>("sintorn2.method.aabbPointer");
    aabbPointer->bindBase(GL_SHADER_STORAGE_BUFFER,5);
  }

  float data[1] = {1.f};
  vars.get<ge::gl::Texture>("shadowMask")->clear(0,GL_RED,GL_FLOAT,data);

  //auto groups = perf::getGroups();
  //for(auto const&g:groups){
  //  std::cerr << g << std::endl;
  //}
  //exit(0);


  prg->use();

  auto const storeTraverseStat = vars.getBool("sintorn2.param.storeTraverseStat");
  if(storeTraverseStat){
    glFinish();
    auto debug = vars.get<Buffer>("sintorn2.method.debug.traverseBuffer");
    glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    debug->bindBase(GL_SHADER_STORAGE_BUFFER,7);
  }


#if SAVE_COLLISION == 1
  debc->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  deb->bindBase(GL_SHADER_STORAGE_BUFFER,5);
  debc->bindBase(GL_SHADER_STORAGE_BUFFER,6);
#endif

  //glFinish();

  //perf::printComputeShaderProf([&]{
  //
  
  if(vars.addOrGetBool("sintorn2.method.perfCounters.rasterize")){
    perf::printComputeShaderProf([&](){
      glDispatchCompute(1024,1,1);
      glMemoryBarrier(GL_ALL_BARRIER_BITS);
    });
  }else{
    glDispatchCompute(1024,1,1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
  }
  //});
  
  //perf::drawFrameWithCounters([&](){
  //  glDispatchCompute(1024,1,1);
  //  glMemoryBarrier(GL_ALL_BARRIER_BITS);
  //});

  //auto mon = perf::Monitor();
  //auto gen = [](uint32_t s){
  //  std::vector<uint32_t>r;
  //  for(uint32_t i=s;i<s+16;++i)
  //    r.push_back(i);
  //  return r;
  //};
  ////2 3 4 6
  ////21 22
  ////34 38 39
  ////48 49 51 55
  ////64 66 67 68 69 70
  ////80 81 84 85 86
  ////102
  //auto cc = gen(16*6);
  //mon.enable("SQ_CS",
  //    //{2,3,4,6,21,22,34,38,39,48,49,51,55,64,66,67});
  //    {2,3,4,6,21,22,34,38,69,70,80,81,84,85,86,102});
  //    //cc);
  //    //{"SQ_CS_000","SQ_CS_001","SQ_CS_002","SQ_CS_003","SQ_CS_004","SQ_CS_005","SQ_CS_006","SQ_CS_007","SQ_CS_008","SQ_CS_009","SQ_CS_010","SQ_CS_011","SQ_CS_012","SQ_CS_013","SQ_CS_014","SQ_CS_015",});
  //    //{"SQ_CS_016","SQ_CS_017","SQ_CS_018","SQ_CS_019","SQ_CS_020","SQ_CS_021","SQ_CS_022","SQ_CS_023","SQ_CS_024","SQ_CS_025","SQ_CS_026","SQ_CS_027","SQ_CS_028","SQ_CS_029","SQ_CS_030","SQ_CS_031",});
  //    //{"SQ_CS_004","SQ_CS_014","SQ_CS_026","SQ_CS_027","SQ_CS_028","SQ_CS_030","SQ_CS_031","SQ_CS_032","SQ_CS_033","SQ_CS_034","SQ_CS_035","SQ_CS_063","SQ_CS_071","SQ_CS_084","SQ_CS_085","SQ_CS_093"});

  //mon.measure([&](){
  //  glDispatchCompute(1024,1,1);
  //  glMemoryBarrier(GL_ALL_BARRIER_BITS);
  //});

  //glFinish();

  //if(storeTraverseStat){
  //  auto debug = vars.get<Buffer>("sintorn2.method.debug.traverseBuffer");
  //  uint32_t NN = 0;
  //  debug->getData(&NN,sizeof(uint32_t));
  //  std::vector<uint32_t>debugData((NN*4+1)*sizeof(uint32_t));
  //  debug->getData(debugData.data(),(NN*4+1)*sizeof(uint32_t));

  //  std::cerr << "N: " << debugData[0] << std::endl;
  //  std::map<uint32_t,uint32_t>levelCnt;
  //  std::map<uint32_t,uint32_t>jobCnt;
  //  std::map<uint32_t,uint32_t>nodeCnt;
  //  std::map<uint32_t,uint32_t>statCnt;
  //  std::map<uint32_t,std::map<uint32_t,uint32_t>>levelStatCnt;
  //  for(uint32_t j=0;j<debugData[0];++j){
  //    auto i = 1+j*4;
  //    auto job   = debugData[i+0];
  //    auto node  = debugData[i+1];
  //    auto level = debugData[i+2];
  //    auto stat  = debugData[i+3];
  //    if(levelCnt.count(level)==0)levelCnt[level] = 0;
  //    if(jobCnt  .count(job  )==0)jobCnt  [job  ] = 0;
  //    if(nodeCnt .count(node )==0)nodeCnt [node ] = 0;
  //    if(statCnt .count(stat )==0)statCnt [stat ] = 0;
  //    if(levelStatCnt.count(level)==0)levelStatCnt[level] = std::map<uint32_t,uint32_t>();
  //    if(levelStatCnt[level].count(stat) == 0)levelStatCnt[level][stat] = 0;
  //    levelCnt[level]++;
  //    jobCnt  [job  ]++;
  //    nodeCnt [node ]++;
  //    statCnt [stat ]++;
  //    levelStatCnt[level][stat]++;
  //  }
  //  for(auto const&x:levelCnt){
  //    std::cerr << "level" << x.first << ": " << x.second << std::endl;
  //  }

  //  //for(auto const&x:jobCnt){
  //  //  std::cerr << "job" << x.first << ": " << x.second << std::endl;
  //  //}

  //  auto statToName = [](uint32_t s){
  //    if(s == 0   )return "empty  ";
  //    if(s == 3   )return "ta     ";
  //    if(s == 2   )return "in     ";
  //    if(s == 0xf0)return "re     ";
  //    if(s == 0xff)return "samples";
  //    return              "bug    ";
  //  };
  //  //for(auto const&x:nodeCnt){
  //  //  std::cerr << "node" << x.first << ": " << x.second << std::endl;
  //  //}
  //  for(auto const&x:statCnt){
  //    std::cerr << "stat " << statToName(x.first) << ": " << x.second << std::endl;
  //  }
  //  for(auto const&l:levelStatCnt){
  //    std::cerr << "level" << l.first << ": " << std::endl;
  //    for(auto const&s:l.second)
  //      std::cerr << "  " << statToName(s.first) << ": " << s.second << std::endl;
  //  }
  //  //exit(0);

  //  //for(uint32_t i=0;i<10000;++i)
  //  //  std::cerr << debugData[i] << " ";
  //  //std::cerr << std::endl;
  //  //auto cfg = *vars.get<Config>("sintorn2.method.config");;
  //  //for(uint32_t j=0;j<debugData[0];++j){
  //  //  auto i = 1+j*3;
  //  //  auto job   = debugData[i+0];
  //  //  auto node  = debugData[i+1];
  //  //  auto level = debugData[i+2];
  //  //  if(level > cfg.nofLevels){
  //  //    std::cerr << "job  : " << job   << " ";
  //  //    std::cerr << "node : " << node  << " ";
  //  //    std::cerr << "level: " << level << std::endl;
  //  //  }
  //  //}
  //}

#if SAVE_COLLISION == 1
  std::vector<float>debData;
  std::vector<uint32_t>debcData;
  debc->getData(debcData);
  //std::cerr << "nn: " << debcData[0] << std::endl;
  deb->getData(debData);
  //for(int i=0;i<debcData[0];++i){
  //  for(int j=0;j<1+3+3+(4+3)*4;++j)
  //    std::cerr << debData[i*(1+3+3+(4+3)*4)+j] << std::endl;
  //  std::cerr << std::endl;
  //}
  //exit(0);
#if 1
  std::map<uint32_t,uint32_t>levelTA;
  for(size_t i=0;i<debcData[0];++i){
    float*dd = debData.data()+i*floatsPerStore;
#define FF std::setprecision(4) << std::fixed << std::showpos

    if((uint32_t)dd[0] == 3u){
      uint32_t o;

      o=1;
      auto level = (uint32_t)dd[o+0];
      if(levelTA.count(level)==0)levelTA[level]=0;
      levelTA[level]++;

      //o=1+1;
      //std::cerr << "aabb: " << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << " - ";
      //o=1+1+3;
      //std::cerr << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << std::endl;
      //o=1+1+3+3;
      //std::cerr << "plane0: " << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << " "<< dd[o+3] << std::endl;
      //o=1+1+3+3+4;
      //std::cerr << "plane1: " << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << " "<< dd[o+3] << std::endl;
      //o=1+1+3+3+4+4;
      //std::cerr << "plane2: " << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << " "<< dd[o+3] << std::endl;
      //o=1+1+3+3+4+4+4;
      //std::cerr << "plane3: " << FF << dd[o+0] << " "<< dd[o+1] << " "<< dd[o+2] << " "<< dd[o+3] << std::endl;
      //std::cerr << std::endl;
    }
  }
  for(auto const&x:levelTA){
    std::cerr << "level" << x.first << ": " << x.second << std::endl;
  }
#endif

#endif
}
