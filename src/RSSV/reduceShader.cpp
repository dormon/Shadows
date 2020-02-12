#include <RSSV/reduceShader.h>

std::string const rssv::reduceShader = R".(
shared float reductionArray[(TILE_X*TILE_Y)*3u];

#if WARP == 32
void reduce(){
  const uint halfWarp        = WARP / 2u;
  const uint halfWarpMask    = uint(halfWarp - 1u);

  float ab[2];
  uint w;

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<3;++k){
  //    float mmin = +1e10;
  //    float mmax = -1e10;
  //    for(uint i=0;i<TILE_X*TILE_Y;++i){
  //      mmin = min(mmin,reductionArray[k*(TILE_X*TILE_Y)+i]);
  //      mmax = max(mmax,reductionArray[k*(TILE_X*TILE_Y)+i]);
  //    }
  //    reductionArray[k*2+0] = mmin;
  //    reductionArray[k*2+1] = mmax;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*0u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*1u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*2u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*3u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);

  ab[0] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*4u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*5u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);
memoryBarrierShared();//even if we have 32 threads WG == warp size of NVIDIA - barrier is necessary on 2080ti


  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<32;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*32+i]);
  //      else
  //        ext = max(ext,reductionArray[k*32+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;


  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<16;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*16+i]);
  //      else
  //        ext = max(ext,reductionArray[k*16+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>3u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>3u)) != 0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<8;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*8+i]);
  //      else
  //        ext = max(ext,reductionArray[k*8+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>4u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();


  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<4;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*4+i]);
  //      else
  //        ext = max(ext,reductionArray[k*4+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>5u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();
 
  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<2;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*2+i]);
  //      else
  //        ext = max(ext,reductionArray[k*2+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>6u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();
}
#endif

#if WARP == 64
void reduce(){
  const uint halfWarp        = WARP / 2u;
  const uint halfWarpMask    = uint(halfWarp - 1u);

  float ab[2];
  uint w;

  ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*0u+gl_LocalInvocationIndex] = ab[w];                         

  ab[0] = reductionArray[WARP*1u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*1u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*1u+gl_LocalInvocationIndex] = ab[w];                         

  ab[0] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];                         



  ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>2u))!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  if(gl_LocalInvocationIndex < (WARP>>1u)){
    ab[0] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>2u))!=0u)) > 0.f);
    reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];
  }



  if((WARP>>3u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>3u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

  if((WARP>>4u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>4u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

  if((WARP>>5u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>5u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }
  
  if((WARP>>6u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>6u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

}
#endif

).";


