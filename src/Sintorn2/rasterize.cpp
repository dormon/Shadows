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
#include <Timer.h>
#include <bitset>
#include <Sintorn2/config.h>

using namespace ge::gl;
using namespace std;

//#define SAVE_COLLISION 1
//#define SAVE_TRAVERSE_STAT 1

#include <string.h>

namespace perf{

std::vector<GLuint>getGroupIDs(){
  GLint numGroups = 0;
  glGetPerfMonitorGroupsAMD(&numGroups,0,nullptr);
  std::vector<GLuint>res(numGroups);
  glGetPerfMonitorGroupsAMD(nullptr, res.size(),res.data());
  return res;
}

std::string getGroupName(GLuint g){
  GLsizei len=0;
  glGetPerfMonitorGroupStringAMD(g,0,&len,nullptr); 
  len++;
  char*buffer = new char[len+1];
  glGetPerfMonitorGroupStringAMD(g,len,nullptr,buffer); 
  auto name = std::string(buffer);
  delete[]buffer;
  return name;
}

std::vector<GLuint>getCounterIDs(GLuint g,GLint&maxActiveCounters){
  GLint numCounters = 0;
  glGetPerfMonitorCountersAMD(g,&numCounters,nullptr,0,nullptr);
  std::vector<GLuint>res(numCounters);
  glGetPerfMonitorCountersAMD(g,&numCounters,&maxActiveCounters,res.size(),res.data());
  return res;
}

std::string getCounterName(GLuint g,GLuint c){
  GLsizei len=0;
  glGetPerfMonitorCounterStringAMD(g,c,0,&len,nullptr);
  len++;
  char*buffer = new char[len];
  glGetPerfMonitorCounterStringAMD(g,c,len,nullptr,buffer);
  auto name = std::string(buffer);
  delete[]buffer;
  return name;
}

class Counter{
  public:
    Counter(GLuint g,GLuint c):id(c),group(g){
      name = getCounterName(g,c);
      glGetPerfMonitorCounterInfoAMD(g,c,GL_COUNTER_TYPE_AMD ,&type);
      if(type == GL_UNSIGNED_INT)
        glGetPerfMonitorCounterInfoAMD(g,c,GL_COUNTER_RANGE_AMD,u32Range);
      if(type == GL_FLOAT)
        glGetPerfMonitorCounterInfoAMD(g,c,GL_COUNTER_RANGE_AMD,f32Range);
      if(type == GL_UNSIGNED_INT64_AMD)
        glGetPerfMonitorCounterInfoAMD(g,c,GL_COUNTER_RANGE_AMD,u64Range);
      if(type == GL_PERCENTAGE_AMD)
        glGetPerfMonitorCounterInfoAMD(g,c,GL_COUNTER_RANGE_AMD,pRange);

    }
    friend std::ostream& operator<< (std::ostream& ss,Counter const& c) {
      ss << c.name << "-" << type2str(c.type);
      return ss;
    }
    static std::string type2str(GLenum type){
      if(type == GL_UNSIGNED_INT      )return "u32";
      if(type == GL_FLOAT             )return "f32";
      if(type == GL_UNSIGNED_INT64_AMD)return "u63";
      if(type == GL_PERCENTAGE_AMD    )return "per";
      return "";
    }
    std::string name;
    GLenum type;
    GLuint id;
    GLuint   u32Range[2];
    GLfloat  f32Range[2];
    GLuint64 u64Range[2];
    GLfloat  pRange  [2];
    GLuint group;
};

std::vector<Counter>getCounters(GLuint g,GLint&maxActiveCounters){
  auto ids = getCounterIDs(g,maxActiveCounters);
  std::vector<Counter>res;
  for(auto const&x:ids)
    res.emplace_back(g,x);
  return res;
}

class Group{
  public:
    Group(GLuint g):id(g){
      name = getGroupName(g);
      counters = getCounters(g,maxActive);
    }
    friend std::ostream& operator<< (std::ostream& ss,Group const& g) {
      ss << g.name << " " << g.maxActive << std::endl;
      for(auto const&c:g.counters)
        ss << "  " << c;
      ss << std::endl;
      return ss;
    }

    GLuint id;
    std::string name;
    GLint maxActive;
    std::vector<Counter>counters;
};

std::vector<Group>getGroups(){
  std::vector<Group>res;
  auto ids = getGroupIDs();
  for(auto const&x:ids)
    res.emplace_back(x);
  return res;
}

class Monitor{
  public:
    Monitor(){
      GLuint id;
      glGenPerfMonitorsAMD(1,&id);
      handle = std::shared_ptr<GLuint>(new GLuint,[](GLuint*i){glDeletePerfMonitorsAMD(1,i);delete i;});
      *handle = id;
      groups = getGroups();
    }
    void enable(std::string const&grpName,std::vector<std::string>const&ctNames){
      GLuint g = 0;
      std::vector<GLuint>cs;
      for(auto const&grp:groups){
        if(grp.name == grpName){
          g = grp.id;
          for(auto const&c:grp.counters){
            for(auto const&nn:ctNames){
              if(nn == c.name){
                cs.push_back(c.id);
              }
            }
          }
        }
      }
      //std::cerr << "g: " << g << std::endl;
      //for(auto const&c:cs)
      //  std::cerr << "  " << c;
      //std::cerr << std::endl;
      glSelectPerfMonitorCountersAMD(*handle,GL_TRUE,g,cs.size(),cs.data());
      //exit(0);
    }
    void enable(std::string const&grpName,std::vector<uint32_t>const&ct){
      GLuint g = 0;
      std::vector<GLuint>cs;
      for(auto const&grp:groups){
        if(grp.name == grpName)
          g = grp.id;
      }
      glSelectPerfMonitorCountersAMD(*handle,GL_TRUE,g,ct.size(),(GLuint*)ct.data());
    }
    void measure(std::function<void()>const&fce){
      glBeginPerfMonitorAMD(*handle);
      fce();
      glEndPerfMonitorAMD(*handle);
      glFinish();
      GLuint dataSize;
      std::cerr << __LINE__ << std::endl;

      uint32_t ready = 0;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_AVAILABLE_AMD,sizeof(ready), &ready,nullptr);
      std::cerr << "ready: " << ready << std::endl;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_SIZE_AMD,sizeof(dataSize), &dataSize,nullptr);
      std::cerr << "dataSize: " << dataSize << std::endl;
      if(dataSize == 0)return;
      std::cerr << __LINE__ << std::endl;
      auto data = std::vector<uint32_t>(dataSize);
      GLsizei bytesWritten;
      std::cerr << __LINE__ << std::endl;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_AMD,dataSize,data.data(), &bytesWritten);
      std::cerr << __LINE__ << std::endl;

      auto name2Name = [&](std::string const&n){
        std::map<std::string,std::string>c2n = {
            {"SQ_CS_004","SQ_CS_PERF_SEL_WAVES"},
            {"SQ_CS_014","SQ_CS_PERF_SEL_ITEMS"},
            {"SQ_CS_026","SQ_CS_PERF_SEL_INSTS_VALU"},
            {"SQ_CS_027","SQ_CS_PERF_SEL_INSTS_VMEM_WR"},
            {"SQ_CS_028","SQ_CS_PERF_SEL_INSTS_VMEM_RD"},
            {"SQ_CS_030","SQ_CS_PERF_SEL_INSTS_SALU"},
            {"SQ_CS_031","SQ_CS_PERF_SEL_INSTS_SMEM"},
            {"SQ_CS_032","SQ_CS_PERF_SEL_INSTS_FLAT"},
            {"SQ_CS_033","SQ_CS_PERF_SEL_INSTS_FLAT"},
            {"SQ_CS_034","SQ_CS_PERF_SEL_INSTS_LDS"},
            {"SQ_CS_035","SQ_CS_PERF_SEL_INSTS_GDS"},
            {"SQ_CS_063","SQ_CS_PERF_SEL_WAIT_INST_LDS"},
            {"SQ_CS_071","SQ_CS_PERF_SEL_ACTIVE_INST_VALU"},
            {"SQ_CS_084","SQ_CS_PERF_SEL_INST_CYCLES_SALU"},
            {"SQ_CS_085","SQ_CS_PERF_SEL_THREAD_CYCLES_VALU"},
            {"SQ_CS_093","SQ_CS_PERF_SEL_LDS_BANK_CONFLICT"},
        };
        if(c2n.count(n) == 0)return n;
        return c2n.at(n);
      };
      GLsizei wordCount = 0;
      while ( (4 * wordCount) < bytesWritten ){
          GLuint groupId = data[wordCount];
          GLuint counterId = data[wordCount + 1];
          auto name = name2Name(getCounterName(groupId,counterId));

          // Determine the counter type
          GLuint counterType;
          glGetPerfMonitorCounterInfoAMD(groupId, counterId,
                                         GL_COUNTER_TYPE_AMD, &counterType);

          if ( counterType == GL_UNSIGNED_INT64_AMD )
          {
            uint64_t value = *(uint64_t*)(&data[wordCount+2]);

            std::cerr << name << " - " << value << std::endl;

            wordCount += 4;
          }
          else if ( counterType == GL_FLOAT )
          {
            float value = *(float*)(&data[wordCount + 2]);

            std::cerr << name << " - " << value << std::endl;

            wordCount += 3;
          }
          else if ( counterType == GL_UNSIGNED_INT){
            uint32_t value = data[wordCount + 2];

            std::cerr << name << " - " << value << std::endl;

            wordCount += 3;
          }
          else if (counterType == GL_PERCENTAGE_AMD){
            float value = *(float*)(&data[wordCount + 2]);

            std::cerr << name << " - " << value << std::endl;

            wordCount += 3;
          }
      }



    }
  protected:
    std::vector<Group>groups;
    std::shared_ptr<GLuint>handle;
};

//////////////////////////////////////////////

typedef struct{
  GLuint* counterList      ;
  int     numCounters      ;
  int     maxActiveCounters;
}CounterInfo;

void getGroupAndCounterList(GLuint **groupsList, int *numGroups,CounterInfo **counterInfo){
  GLint          n;
  GLuint        *groups;
  CounterInfo   *counters;

  glGetPerfMonitorGroupsAMD(&n, 0, NULL);
  groups = (GLuint*) malloc(n * sizeof(GLuint));
  glGetPerfMonitorGroupsAMD(NULL, n, groups);
  *numGroups = n;

  *groupsList = groups;
  counters = (CounterInfo*) malloc(sizeof(CounterInfo) * n);
  for (int i = 0 ; i < n; i++ )
  {
      glGetPerfMonitorCountersAMD(groups[i], &counters[i].numCounters,
                               &counters[i].maxActiveCounters, 0, NULL);

      counters[i].counterList = (GLuint*)malloc(counters[i].numCounters *
                                                sizeof(int));

      glGetPerfMonitorCountersAMD(groups[i], NULL, NULL,
                                  counters[i].numCounters,
                                  counters[i].counterList);
  }

  *counterInfo = counters;
}

static int  countersInitialized = 0;

int getCounterByName(char const*groupName, char const*counterName, GLuint *groupID,GLuint *counterID){
  int          numGroups;
  GLuint       *groups;
  CounterInfo  *counters;
  int          i = 0;

  if (!countersInitialized)
  {
      getGroupAndCounterList(&groups, &numGroups, &counters);
      countersInitialized = 1;
  }

  for ( i = 0; i < numGroups; i++ )
  {
     char curGroupName[256];
     glGetPerfMonitorGroupStringAMD(groups[i], 256, NULL, curGroupName);
     if (strcmp(groupName, curGroupName) == 0)
     {
         *groupID = groups[i];
         break;
     }
  }

  if ( i == numGroups )
      return -1;           // error - could not find the group name

  for ( int j = 0; j < counters[i].numCounters; j++ )
  {
      char curCounterName[256];

      glGetPerfMonitorCounterStringAMD(groups[i],
                                       counters[i].counterList[j],
                                       256, NULL, curCounterName);
      if (strcmp(counterName, curCounterName) == 0)
      {
          *counterID = counters[i].counterList[j];
          return 0;
      }
  }

  return -1;           // error - could not find the counter name
}

void drawFrameWithCounters(std::function<void()>const&fce){
  GLuint group[2];
  GLuint counter[2];
  GLuint monitor;
  GLuint *counterData;

  // Get group/counter IDs by name.  Note that normally the
  // counter and group names need to be queried for because
  // each implementation of this extension on different hardware
  // could define different names and groups.  This is just provided
  // to demonstrate the API.
  getCounterByName("HW", "Hardware Busy", &group[0],
                   &counter[0]);
  getCounterByName("API", "Draw Calls", &group[1],
                   &counter[1]);

  // create perf monitor ID
  glGenPerfMonitorsAMD(1, &monitor);

  // enable the counters
  glSelectPerfMonitorCountersAMD(monitor, GL_TRUE, group[0], 1,
                                 &counter[0]);
  glSelectPerfMonitorCountersAMD(monitor, GL_TRUE, group[1], 1,
                                 &counter[1]);

  glBeginPerfMonitorAMD(monitor);

  fce();

  // RENDER FRAME HERE
  // ...

  glEndPerfMonitorAMD(monitor);

  // read the counters
  GLuint resultSize;
  glGetPerfMonitorCounterDataAMD(monitor, GL_PERFMON_RESULT_SIZE_AMD,
                                 sizeof(GLint), &resultSize, NULL);


  counterData = (GLuint*) malloc(resultSize);

  GLsizei bytesWritten;
  glGetPerfMonitorCounterDataAMD(monitor, GL_PERFMON_RESULT_AMD,
                                 resultSize, counterData, &bytesWritten);

  // display or log counter info
  GLsizei wordCount = 0;

  while ( (4 * wordCount) < bytesWritten )
  {
      GLuint groupId = counterData[wordCount];
      GLuint counterId = counterData[wordCount + 1];

      // Determine the counter type
      GLuint counterType;
      glGetPerfMonitorCounterInfoAMD(groupId, counterId,
                                     GL_COUNTER_TYPE_AMD, &counterType);

      if ( counterType == GL_UNSIGNED_INT64_AMD )
      {
          uint64_t counterResult =
                     *(uint64_t*)(&counterData[wordCount + 2]);

          std::cerr << "A uint64_t: " << counterResult << std::endl;
          // Print counter result

          wordCount += 4;
      }
      else if ( counterType == GL_FLOAT )
      {
          float counterResult = *(float*)(&counterData[wordCount + 2]);

          std::cerr << "A float: " << counterResult;

          // Print counter result

          wordCount += 3;
      }
      // else if ( ... ) check for other counter types
      //   (GL_UNSIGNED_INT and GL_PERCENTAGE_AMD)
  }
}

}


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
#if SAVE_COLLISION == 1
        Shader::define("SAVE_COLLISION"     ,(int)1),
#endif
#if SAVE_TRAVERSE_STAT == 1
        Shader::define("SAVE_TRAVERSE_STAT" ,(int)1),
#endif
        ballotSrc,
        sintorn2::demortonShader,
        sintorn2::configShader,
        sintorn2::rasterizeShader
        ));

}

#if SAVE_TRAVERSE_STAT == 1
std::shared_ptr<Buffer>debug;
#endif

#if SAVE_COLLISION == 1
std::shared_ptr<Buffer>deb;
std::shared_ptr<Buffer>debc;
const uint32_t floatsPerStore = 1 + 1 + 3 + 3 + (4+3)*4;
#endif

void createJobCounter(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method");
  vars.reCreate<Buffer>("sintorn2.method.jobCounter",sizeof(uint32_t));

#if SAVE_TRAVERSE_STAT == 1
  debug = make_shared<Buffer>(sizeof(uint32_t)*(1+1024*1024*128));
#endif

#if SAVE_COLLISION == 1
  deb = make_shared<Buffer>(sizeof(float)*floatsPerStore*10000);
  debc = make_shared<Buffer>(sizeof(uint32_t)*(1));
#endif
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

  //auto groups = perf::getGroups();
  //for(auto const&g:groups){
  //  std::cerr << g << std::endl;
  //}
  //exit(0);


  prg->use();

#if SAVE_TRAVERSE_STAT == 1
  glFinish();
  //debug->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  glClearNamedBufferSubData(debug->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
  debug->bindBase(GL_SHADER_STORAGE_BUFFER,7);
#endif

#if SAVE_COLLISION == 1
  debc->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  deb->bindBase(GL_SHADER_STORAGE_BUFFER,5);
  debc->bindBase(GL_SHADER_STORAGE_BUFFER,6);
#endif

  //glFinish();

  glDispatchCompute(1024,1,1);
  glMemoryBarrier(GL_ALL_BARRIER_BITS);
  
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

#if SAVE_TRAVERSE_STAT == 1
  uint32_t NN = 0;
  debug->getData(&NN,sizeof(uint32_t));
  std::vector<uint32_t>debugData((NN*4+1)*sizeof(uint32_t));
  debug->getData(debugData.data(),(NN*4+1)*sizeof(uint32_t));

  std::cerr << "N: " << debugData[0] << std::endl;
  std::map<uint32_t,uint32_t>levelCnt;
  std::map<uint32_t,uint32_t>jobCnt;
  std::map<uint32_t,uint32_t>nodeCnt;
  std::map<uint32_t,uint32_t>statCnt;
  std::map<uint32_t,std::map<uint32_t,uint32_t>>levelStatCnt;
  for(uint32_t j=0;j<debugData[0];++j){
    auto i = 1+j*4;
    auto job   = debugData[i+0];
    auto node  = debugData[i+1];
    auto level = debugData[i+2];
    auto stat  = debugData[i+3];
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

  auto statToName = [](uint32_t s){
    if(s == 0   )return "empty  ";
    if(s == 3   )return "ta     ";
    if(s == 2   )return "in     ";
    if(s == 0xf0)return "re     ";
    if(s == 0xff)return "samples";
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
  //auto cfg = *vars.get<Config>("sintorn2.method.config");;
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
#endif

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
