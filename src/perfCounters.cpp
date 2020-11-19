#include<vector>
#include<ostream>
#include<map>
#include<iostream>

#include <geGL/StaticCalls.h>

#include <perfCounters.h>

using namespace ge::gl;

namespace perf{

std::vector<GLuint>getGroupIDs(){
  GLint numGroups = 0;
  glGetPerfMonitorGroupsAMD(&numGroups,0,nullptr);
  std::vector<GLuint>res(numGroups);
  glGetPerfMonitorGroupsAMD(nullptr, GLsizei(res.size()),res.data());
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
  glGetPerfMonitorCountersAMD(g,&numCounters,&maxActiveCounters, GLsizei(res.size()),res.data());
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
      glSelectPerfMonitorCountersAMD(*handle,GL_TRUE,g, GLsizei(cs.size()),cs.data());
      //exit(0);
    }
    void enable(std::string const&grpName,std::vector<uint32_t>const&ct){
      GLuint g = 0;
      std::vector<GLuint>cs;
      for(auto const&grp:groups){
        if(grp.name == grpName)
          g = grp.id;
      }
      glSelectPerfMonitorCountersAMD(*handle,GL_TRUE,g, GLsizei(ct.size()),(GLuint*)ct.data());
    }
    void measure(std::function<void()>const&fce){
      glBeginPerfMonitorAMD(*handle);
      fce();
      glEndPerfMonitorAMD(*handle);
      glFinish();
      GLuint dataSize;

      uint32_t ready = 0;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_AVAILABLE_AMD,sizeof(ready), &ready,nullptr);
      if(!ready)return;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_SIZE_AMD,sizeof(dataSize), &dataSize,nullptr);
      if(dataSize == 0)return;
      auto data = std::vector<uint32_t>(dataSize);
      GLsizei bytesWritten;
      glGetPerfMonitorCounterDataAMD(*handle, GL_PERFMON_RESULT_AMD,dataSize,data.data(), &bytesWritten);

      auto name2Name = [&](std::string const&n){
        std::map<std::string,std::string>c2n = {
            {"SQ_CS_004","SQ_CS_PERF_SEL_WAVES"             },
            {"SQ_CS_014","SQ_CS_PERF_SEL_ITEMS"             },
            {"SQ_CS_026","SQ_CS_PERF_SEL_INSTS_VALU"        },
            {"SQ_CS_027","SQ_CS_PERF_SEL_INSTS_VMEM_WR"     },
            {"SQ_CS_028","SQ_CS_PERF_SEL_INSTS_VMEM_RD"     },
            {"SQ_CS_030","SQ_CS_PERF_SEL_INSTS_SALU"        },
            {"SQ_CS_031","SQ_CS_PERF_SEL_INSTS_SMEM"        },
            {"SQ_CS_032","SQ_CS_PERF_SEL_INSTS_FLAT"        },
            {"SQ_CS_033","SQ_CS_PERF_SEL_INSTS_FLAT"        },
            {"SQ_CS_034","SQ_CS_PERF_SEL_INSTS_LDS"         },
            {"SQ_CS_035","SQ_CS_PERF_SEL_INSTS_GDS"         },
            {"SQ_CS_061","SQ_CS_PERF_SEL_WAIT_INST_LDS"     },
            {"SQ_CS_081","SQ_CS_PERF_SEL_ACTIVE_INST_VALU"  },
            {"SQ_CS_086","SQ_CS_PERF_SEL_INST_CYCLES_SALU"  },
            {"SQ_CS_089","SQ_CS_PERF_SEL_THREAD_CYCLES_VALU"},
            {"SQ_CS_096","SQ_CS_PERF_SEL_LDS_BANK_CONFLICT" },

            {"SQ_LS_004","SQ_LS_PERF_SEL_WAVES"              },
            {"SQ_LS_014","SQ_LS_PERF_SEL_ITEMS"              },
            {"SQ_LS_026","SQ_LS_PERF_SEL_INSTS_VALU"         },
            {"SQ_LS_027","SQ_LS_PERF_SEL_INSTS_VMEM_WR"      },
            {"SQ_LS_028","SQ_LS_PERF_SEL_INSTS_VMEM_RD"      },
            {"SQ_LS_030","SQ_LS_PERF_SEL_INSTS_SALU"         },
            {"SQ_LS_031","SQ_LS_PERF_SEL_INSTS_SMEM"         },
            {"SQ_LS_032","SQ_LS_PERF_SEL_INSTS_FLAT"         },
            {"SQ_LS_033","SQ_LS_PERF_SEL_INSTS_FLAT_LDS_ONLY"},
            {"SQ_LS_034","SQ_LS_PERF_SEL_INSTS_LDS"          },
            {"SQ_LS_035","SQ_LS_PERF_SEL_INSTS_GDS"          },
            {"SQ_LS_061","SQ_LS_PERF_SEL_WAIT_INST_LDS"      },
            {"SQ_LS_081","SQ_LS_PERF_SEL_INST_CYCLES_VALU"   },
            {"SQ_LS_086","SQ_LS_PERF_SEL_INST_CYCLES_SALU"   },
            {"SQ_LS_089","SQ_LS_PERF_SEL_THREAD_CYCLES_VALU" },
            {"SQ_LS_097","SQ_LS_PERF_SEL_LDS_BANK_CONFLICT"  },
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

void printComputeShaderProf(std::function<void()>const&fce){
  Monitor mon;

  auto gen = [](uint32_t s){
    std::vector<uint32_t>r;
    for(uint32_t i=s;i<s+16;++i)
      r.push_back(i);
    return r;
  };
  ////2 3 4 6
  ////21 22
  ////34 38 39
  ////48 49 51 55
  ////64 66 67 68 69 70
  ////80 81 84 85 86
  ////102
  auto cc = gen(16*0);

  /*
  mon.enable("SQ_LS",{
      4,14,26,27,28,30,31,32,33,34,35,61,81,86,89,97,
      }
      );
  // */


//*
  mon.enable("SQ_CS",
#if 0
      //cc
#else    
      {
      "SQ_CS_004",
      "SQ_CS_014",
      "SQ_CS_026",
      "SQ_CS_027",
      "SQ_CS_028",
      "SQ_CS_030",
      "SQ_CS_031",
      "SQ_CS_032",
      "SQ_CS_033",
      "SQ_CS_034",
      "SQ_CS_035",
      "SQ_CS_061",
      "SQ_CS_081",
      "SQ_CS_086",
      "SQ_CS_089",
      "SQ_CS_096",
      }

      );
#endif

// */

  mon.measure(fce);
}

void printComputeShaderProf(std::function<void()>const&fce,uint32_t counter){
  Monitor mon;
  mon.enable("SQ_CS",std::vector<GLuint>({counter}));
  mon.measure(fce);
}

}
