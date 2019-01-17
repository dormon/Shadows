#include<string>

std::string const ballotSrc = R".(

#extension GL_ARB_gpu_shader_int64 : enable
#if !defined(GL_ARB_gpu_shader_int64)
  #extension GL_AMD_gpu_shader_int64 : enable
  //#if !defined(GL_AMD_gpu_shader_int64)
  //#endif
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
      #define TRANSFORM_BALLOT_RESULT_TO_UINTS(result) result
      #define UINT_RESULT_ARRAY                        uint
      #define GET_UINT_FROM_UINT_ARRAY(array,i)        array
      #define BALLOT_RESULT_LENGTH                     1
    #endif
  #else
    #define BALLOT(x)                                ballotAMD(x)
    #define TRANSFORM_BALLOT_RESULT_TO_UINTS(result) unpackUint2x32(result)
    #define UINT_RESULT_ARRAY                        uvec2
    #define GET_UINT_FROM_UINT_ARRAY(array,i)        array[i]
    #define BALLOT_RESULT_LENGTH                     2
  #endif
#else
#define BALLOT(x)                                ballotARB(x)
#define TRANSFORM_BALLOT_RESULT_TO_UINTS(result) unpackUint2x32(result)
#define UINT_RESULT_ARRAY                        uvec2
#define GET_UINT_FROM_UINT_ARRAY(array,i)        array[i]
#define BALLOT_RESULT_LENGTH                     2
#endif

//#if WAVEFRONT_SIZE == 64
//  #extension GL_AMD_gcn_shader       : enable
//  #extension GL_AMD_gpu_shader_int64 : enable
//  #define BALLOT(x)                                ballotAMD(x)
//  #define TRANSFORM_BALLOT_RESULT_TO_UINTS(result) unpackUint2x32(result)
//  #define UINT_RESULT_ARRAY                        uvec2
//  #define GET_UINT_FROM_UINT_ARRAY(array,i)        array[i]
//  #define BALLOT_RESULT_LENGTH                     2
//#else
//  #extension GL_NV_shader_thread_group : enable
//  #define BALLOT(x)                                ballotThreadNV(x)
//  #define TRANSFORM_BALLOT_RESULT_TO_UINTS(result) result
//  #define UINT_RESULT_ARRAY                        uint
//  #define GET_UINT_FROM_UINT_ARRAY(array,i)        array
//  #define BALLOT_RESULT_LENGTH                     1
//#endif

).";
