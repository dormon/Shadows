#include <BallotShader.h>

std::string const ballotSrc = R".(

#extension GL_ARB_gpu_shader_int64 : enable
#if !defined(GL_ARB_gpu_shader_int64)
  #extension GL_AMD_gpu_shader_int64 : enable
#endif

#extension GL_ARB_shader_ballot : enable
#if !defined(GL_ARB_shader_ballot)
  #extension GL_AMD_gcn_shader : enable
  #if !defined(GL_AMD_gcn_shader)
    #extension GL_NV_shader_thread_group : enable
    #if !defined(GL_NV_shader_thread_group)
      #error "Ballot is unsupported!"
    #else
      #define BALLOT(x)                                ballotThreadNV(x)
      #define BALLOT_RESULT_TO_UINTS(result)           result
      #define BALLOT_RESULT                            uint
      #define BALLOT_UINTS                             uint
      #define BALLOT_LENGTH                            1
      #define GET_UINT_FROM_UINT_ARRAY(array,i)        array
    #endif
  #else
    #define BALLOT(x)                                ballotAMD(x)
    #define BALLOT_RESULT_TO_UINTS(result)           unpackUint2x32(result)
    #define BALLOT_RESULT                            uint64_t
    #define BALLOT_UINTS                             uvec2
    #define BALLOT_LENGTH                            2
    #define GET_UINT_FROM_UINT_ARRAY(array,i)        array[i]
  #endif
#else
  #define BALLOT(x)                                ballotARB(x)
  #define BALLOT_RESULT_TO_UINTS(result)           unpackUint2x32(result)
  #define BALLOT_RESULT                            uint64_t
  #define BALLOT_UINTS                             uvec2
  #define BALLOT_LENGTH                            (gl_SubGroupSizeARB >> 5)
  #define GET_UINT_FROM_UINT_ARRAY(array,i)        array[i]
#endif


).";
