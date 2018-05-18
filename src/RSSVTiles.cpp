#include<RSSVTiles.h>
#include<algorithm>
#include<util.h>

// threadsPerTile         - number of threads per tile - warp/wavefront size
// threadsExponent        - threadsPerTile = 2^threadsExponent
// windowSize             - size of window in pixels
// windowExponent         - 2^windowExponent <= windowSize < 2^(windowExponent+1)
// compoundWindowExponent - windowExponent.x + windowExponent.y
// nofLevels              - number of levels of image pyramid
//                          nofLevels = roundUp( compoundWindowExponent / threadsExponent )
//
// Each level decreases compoundWindowExponent by threadsExponent.
// threadsExponent is divided into two nonzero numbers:
// threadsExponentX(level) + threadsExponentY(level) = threadsExponent
// threadsExponentX(level) decreases widthExponent
// threadsExponentY(level) decreases heightExponent
//
// accumulatedExponentX(level) = threadsExponentX(level) + ... + threadsExponentX(nofLevels-1)
// accumulatedExponentY(level) = threadsExponentY(level) + ... + threadsExponentY(nofLevels-1)
//
// threadsExponentX(level) and threadsExponentY(level) is chosen as follows:
//
// 1. accumulatedExponentX(level) and accumulatedExponentY(level) differes 
//    from each other for all levels that are no zero level by 1 at most.
// 2. accumulatedExponentX(0) == widthExponent ||
//    accumulatedExponentY(0) == heightExponent
//    One axis has to have accumulatedExponent(0) == windowExponent
//
// FULL___TILE_SIZE_IN__TILES
// FULL___TILE_SIZE_IN_PIXELS
// BORDER_TILE_SIZE_IN__TILES
// BORDER_TILE_SIZE_IN_PIXELS
//
// NOF______TILES_PER_LEVEL(level)
// NOF_FULL_TILES_PER_LEVEL(level) 
//
// 1011 x 511, 32
//
// 4x8 1024 x 1024   
// 8x4  256 x  128
// 4x8   32 x   32
// 8x4    8 x    4 
//
// An image is a tile.
// A tile has resolution in pixels.
// A tile is composed of full tiles and border tiles.
// An image with resolution 1011 x 511 and threads 32:
// Image tile size in pixel is 1011 x 511.
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(1024,1024) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(4   ,8   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(1011,511 )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(4   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(0) uvec2(3   ,3   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(256 ,128 )
// FULL___TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(8   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(243 ,127 )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(8   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(1) uvec2(7   ,3   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(32  ,32  )
// FULL___TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(4   ,8   )
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(19  ,31  )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(3   ,8   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(2) uvec2(2   ,7   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(8   ,4   )
// FULL___TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(8   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(3   ,3   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(3   ,3   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(3) uvec2(3   ,3   )
//
// Image tile size in pixels is 1037 x 111, 32 threads
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(8192,128 ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(32  ,1   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(1037,111 )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(5   ,1   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(0) uvec2(4   ,0   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(256 ,128 ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(8   ,4   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(13  ,111 )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(1   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(1) uvec2(0   ,3   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(32  ,32  ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(4   ,8   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(13  ,15  )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(2   ,4   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(2) uvec2(1   ,3   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(8   ,4   ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(8   ,4   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(5   ,3   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(5   ,3   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(3) uvec2(5   ,3   )
//
// Image tile size in pixels is 1024 x 1024, 32 threads
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(1024,1024) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(4   ,8   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (0) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (0) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(0) uvec2(0   ,0   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(256 ,128 ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(8   ,4   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (1) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (1) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(1) uvec2(0   ,0   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(32  ,32  ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(4   ,8   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (2) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (2) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(2) uvec2(0   ,0   )
//
// FULL___TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(8   ,4   ) 
// FULL___TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(8   ,4   ) 
// BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (3) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_TILES     (3) uvec2(0   ,0   )
// BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(3) uvec2(0   ,0   )
//
//
//
// FULL_TILE_SIZE_IN_PIXELS(1)   uvec2(256,128)
// BORDER_TILE_SIZE_IN_PIXELS(1) uvec2(243,127)
// TILE_SIZE_IN_TILES (0) 
//
//
// 
// LEVEL_DIVISIBILITY                (level) (3,1) no zero possible
// LEVEL_DIVISIBILITY_INTO_FULL_TILES(level) (2,0) zero possible
// LEVEL_CONFIGURATION               (level)
// FULL_TILE_SIZE_IN_TILES           (level) (4,8) multiplication of components has to be warp
// BORDER_TILE_SIZE_IN_TILES         (level) (3,5) multiplication of components has to be <= warp
// FULL_TILE_SIZE_IN_PIXELS          (level) (512,512)
// BORDER_TILE_SIZE_IN_PIXELS        (level) (387,277)
//

class RSSVTilingImpl{
  public:
    RSSVTilingImpl(size_t windowWidth,size_t windowHeight,size_t threadsPerTile){
      windowSize.x = windowWidth ;
      windowSize.y = windowHeight;

      windowExponent.x = log2RoundUp(windowSize.x  );
      windowExponent.y = log2RoundUp(windowSize.y  );
      threadsExponent  = log2RoundUp(threadsPerTile);

      compoundWindowExponent = windowExponent.x + windowExponent.y;

      nofLevels = divRoundUp<size_t>(compoundWindowExponent,threadsExponent);

      fullTileSizeInTiles.resize(nofLevels,glm::uvec2(0));

      auto exponentCounter = glm::uvec2(0u);

      size_t const threadsExponentPart[2] = {
                          divRoundUp(threadsExponent,2lu),
        threadsExponent - divRoundUp(threadsExponent,2lu),
      };

      bool oddLevel = false;
      for(auto&x:fullTileSizeInTiles){
        glm::uvec2 currentExponent;
        currentExponent.x = threadsExponentPart[  oddLevel];
        currentExponent.y = threadsExponentPart[1-oddLevel];

        if      (exponentCounter[0] + currentExponent[0] > windowExponent[0]){
          currentExponent[0] = windowExponent[0] - exponentCounter[0];
          currentExponent[1] = threadsExponent   - currentExponent[0];
        }else if(exponentCounter[1] + currentExponent[1] > windowExponent[1]){
          currentExponent[1] = windowExponent[1] - exponentCounter[1];
          currentExponent[0] = threadsExponent   - currentExponent[1];
        }

        x.x = 1 << currentExponent.x;
        x.y = 1 << currentExponent.y;

        exponentCounter += currentExponent;
        oddLevel = !oddLevel;
      }
      std::reverse(fullTileSizeInTiles.begin(),fullTileSizeInTiles.end());
    }
    size_t     nofLevels             ;
    size_t     threadsExponent       ;
    size_t     compoundWindowExponent;
    glm::uvec2 windowSize            ;
    glm::uvec2 windowExponent        ;
    std::vector<glm::uvec2>fullTileSizeInTiles;

};

RSSVTiling::RSSVTiling(size_t windowWidth,size_t windowHeight,size_t threadsPerTile){
  _impl = std::unique_ptr<RSSVTilingImpl>(new RSSVTilingImpl(windowWidth,windowHeight,threadsPerTile));
}


size_t                 RSSVTiling::getNofLevels()const{
  return _impl->nofLevels;
}

glm::uvec2             RSSVTiling::getWindowSize()const{
  return _impl->windowSize;
}


glm::uvec2             RSSVTiling::getWindowExponent                ()const{
  return _impl->windowExponent;
}
/*
   std::vector<glm::uvec2>RSSVTilin::getLevelDivisibility             ()const;
   std::vector<glm::uvec2>RSSVTilin::getLevelDivisibilityIntoFullTiles()const;
   std::vector<glm::uvec2>RSSVTilin::getFullTileSizeInTiles           ()const;
   std::vector<glm::uvec2>RSSVTilin::getFullTileSizeInPixels          ()const;
   std::vector<glm::uvec2>RSSVTilin::getBorderTileSizeInTiles         ()const;
   std::vector<glm::uvec2>RSSVTilin::getBorderTileSizeInPixels        ()const;
   */
size_t rssvGetNofLevels(
    glm::uvec2 const&windowSize    ,
    size_t     const&threadsPerTile){
  auto const widthExponent   = log2RoundUp(windowSize.x  );
  auto const heightExponet   = log2RoundUp(windowSize.y  );
  auto const threadsExponent = log2RoundUp(threadsPerTile);
  return divRoundUp<size_t>(widthExponent + heightExponet,threadsExponent);
}


std::vector<glm::uvec2>rssvGetMaxUpperTileDivisibility(
    glm::uvec2 const&windowSize    ,
    size_t     const&threadsPerTile){
  size_t const nofLevels = rssvGetNofLevels(windowSize,threadsPerTile);
  auto const threadsExponent = log2RoundUp(threadsPerTile);
  std::vector<glm::uvec2>result;
  result.resize(nofLevels,glm::uvec2(0));
  //NVIDIA has WARP = 32 threads
  //8x4 for every even level
  //4x8 for every odd level
  //
  //AMD has WARP = 64 threads
  //8x8 for every level
  size_t const threadsExponentPart[2] = {
    divRoundUp(threadsExponent,2lu),
    threadsExponent - divRoundUp(threadsExponent,2lu),
  };
  auto const windowExponent = glm::uvec2(
      log2RoundUp(windowSize.x),
      log2RoundUp(windowSize.y));

  auto exponentCounter = glm::uvec2(0u);

  bool oddLevel = false;
  for(auto&x:result){
    glm::uvec2 currentExponent;
    currentExponent.x = threadsExponentPart[  oddLevel];
    currentExponent.y = threadsExponentPart[1-oddLevel];

    for(size_t i=0;i<2;++i)
      if(exponentCounter[i] + currentExponent[i] > windowExponent[i]){
        currentExponent[  i] = windowExponent[i] - exponentCounter[i];
        currentExponent[1-i] = threadsExponent - currentExponent[i];
        if(exponentCounter[1-i] + currentExponent[1-i] > windowExponent[1-i])
          currentExponent[1-i] = windowExponent[1-i] - exponentCounter[1-i];
      }

    x.x = 1 << currentExponent.x;
    x.y = 1 << currentExponent.y;

    exponentCounter += currentExponent;
    oddLevel = !oddLevel;
  }
  std::reverse(result.begin(),result.end());
  return result;
}

class RSSVTilesSolution{
  public:
    std::vector<glm::uvec2>levels;
    size_t idle = 0;
    size_t squareness = 0;
    void computeSquareness(){
      assert(this!=nullptr);
      size_t wgs = this->levels.front().x*this->levels.front().y;
      auto ii=this->levels.rbegin();
      glm::uvec2 curSize = *ii;
      this->squareness=wgs/curSize.x;
      ii++;
      curSize = curSize**ii-glm::uvec2(ii->x,0);
      this->squareness*=2;
      if(curSize.x<curSize.y)this->squareness+=curSize.y/curSize.x;
      else this->squareness+=curSize.x/curSize.y;
      ii++;
      for(;ii!=this->levels.rend();ii++){
        curSize*=*ii;
        this->squareness*=2;
        if(curSize.x<curSize.y)this->squareness+=curSize.y/curSize.x;
        else this->squareness+=curSize.x/curSize.y;
      }
    }
    void computeIdle(glm::uvec2 const&windowSize){
      assert(this!=nullptr);
      this->idle = 0;
      glm::uvec2 prevSize = windowSize;
      int i=int(this->levels.size()-1);
      this->idle += 
        divRoundUp(prevSize.x,this->levels.at(i).x-1)*(this->levels.at(i).x-1)*
        divRoundUp(prevSize.y,this->levels.at(i).y  )*(this->levels.at(i).y  )-
        prevSize.x*prevSize.y;
      prevSize.x = uint32_t(divRoundUp(prevSize.x,this->levels.at(i).x-1));
      prevSize.y = uint32_t(divRoundUp(prevSize.y,this->levels.at(i).y  ));
      i--;
      for(;i>=0;--i){
        this->idle += 
          divRoundUp(prevSize.x,this->levels.at(i).x)*(this->levels.at(i).x)*
          divRoundUp(prevSize.y,this->levels.at(i).y)*(this->levels.at(i).y)-
          prevSize.x*prevSize.y;
        prevSize.x = uint32_t(divRoundUp(prevSize.x,this->levels.at(i).x));
        prevSize.y = uint32_t(divRoundUp(prevSize.y,this->levels.at(i).y));
      }
    }
    glm::uvec2 getSize()const{
      assert(this!=nullptr);
      if(this->levels.size()==0)return glm::uvec2(0);
      auto ii=this->levels.rbegin();
      glm::uvec2 curSize = *ii;
      ii++;
      bool joiningLevelHapped = false;
      for(;ii!=this->levels.rend();ii++){
        if(joiningLevelHapped){
          curSize*=*ii;
        }else{
          if(ii->x==1){
            curSize*=*ii;
          }else{
            curSize = curSize**ii-glm::uvec2(ii->x,0);
            joiningLevelHapped = true;
          }
        }
      }
      return curSize;
    }
    bool operator<(RSSVTilesSolution const&other)const{
      assert(this!=nullptr);
      if(this->levels.size()<other.levels.size())return true;
      if(this->levels.size()>other.levels.size())return false;
      if(this->levels.back().x>other.levels.back().x)return true;
      if(this->levels.back().x<other.levels.back().x)return false;
      if(this->idle<other.idle)return true;
      if(this->idle>other.idle)return false;
      return this->squareness<other.squareness;
    }
};

void rssvTileSizeChoises(
    std::vector<glm::uvec2>&choices,
    size_t threadsPerTile){
  for(size_t x=1;x<=threadsPerTile;++x)//loop over all choices
    if((threadsPerTile%x)==0)//is this choice of x possible?
      choices.push_back(glm::uvec2(x,threadsPerTile/x));
}

size_t rssvComputeNofLevels(glm::uvec2 const&windowSize,size_t threadsPerTile){
  return (size_t)std::ceil(glm::log(windowSize[0]*windowSize[1])/glm::log(threadsPerTile))+1;
}

void rssvGenerateSolutions(
    std::vector<RSSVTilesSolution>&solutions,
    glm::uvec2 const&windowSize,
    size_t threadsPerTile){
  std::vector<glm::uvec2>choices;
  rssvTileSizeChoises(choices,threadsPerTile);
  size_t nofLevels = rssvComputeNofLevels(windowSize,threadsPerTile);
  size_t II=0;//index into index
  std::vector<size_t>index;
  index.resize(nofLevels,0);

  size_t index1D=0;//1D version of index
  do{//loop over all solutions
    RSSVTilesSolution sol;
    for(size_t i=0;i<nofLevels;++i){
      if( sol.getSize().x>=windowSize.x &&
          sol.getSize().y>=windowSize.y)break;
      sol.levels.push_back(choices[index[i]]);
    }
    if(sol.levels.back().x>1){
      sol.computeIdle(windowSize);
      sol.computeSquareness();
      if( sol.getSize().x>=windowSize.x &&
          sol.getSize().y>=windowSize.y)
        solutions.push_back(sol);
    }
    ++index1D;//increment 1D version of index
    //increment index to solutions
    II=0;
    do{//loop over levels
      ++index[II];//increment index in II level
      if(index[II]>=choices.size()){//index at II level overflows
        index[II]=0;//clear index in II level
        ++II;//increment level
      }else break;//we are done incrementing
    }while(II<nofLevels);
  }while(II<nofLevels);
}

void rssvChooseTileSizes(
    std::vector<glm::uvec2>&tileDivisibility,
    glm::uvec2 const&windowSize,
    size_t threadsPerTile){
  std::vector<RSSVTilesSolution>solutions;
  rssvGenerateSolutions(solutions,windowSize,threadsPerTile);
  std::sort(solutions.begin(),solutions.end());
  for(auto const&x:solutions.front().levels)
    tileDivisibility.push_back(x);
}


RSSVTilingSizes::RSSVTilingSizes(
    glm::uvec2 const&windowSize    ,
    size_t     const&threadsPerTile){
  auto const nofLevels       = rssvGetNofLevels(windowSize,threadsPerTile);
  auto const threadsExponent = log2RoundUp     (threadsPerTile);

  size_t const threadsExponentPart[2] = {
                      divRoundUp(threadsExponent,2lu),
    threadsExponent - divRoundUp(threadsExponent,2lu),
  };

  auto const windowExponent = glm::uvec2(
      log2RoundUp(windowSize.x),
      log2RoundUp(windowSize.y));


  full__TileDivisibilityIntoPixels   .resize(nofLevels);
  full__TileDivisibilityIntoTiles    .resize(nofLevels);
  borderTileDivisibilityIntoPixels   .resize(nofLevels);
  borderTileDivisibilityIntoTiles    .resize(nofLevels);
  borderTileDivisibilityIntoFullTiles.resize(nofLevels);


  int64_t const lastLevel  = nofLevels - 1;
  int64_t const firstLevel = 0            ;


  auto exponentCounter = glm::uvec2(0u);
  bool oddLevel = false;
  for(int64_t level = lastLevel; level >= firstLevel; --level){

    glm::uvec2 currentExponent;
    currentExponent.x = threadsExponentPart[  static_cast<size_t>(oddLevel)];
    currentExponent.y = threadsExponentPart[1-static_cast<size_t>(oddLevel)];

    if(exponentCounter[0] + currentExponent[0] > windowExponent[0]){
      currentExponent[0] = windowExponent[0] - exponentCounter[0];
      currentExponent[1] = threadsExponent - currentExponent[0];
    }else{
      if(exponentCounter[1] + currentExponent[1] > windowExponent[1]){
        currentExponent[1] = windowExponent[1] - exponentCounter[1];
        currentExponent[0] = threadsExponent - currentExponent[1];
      }
    }

    full__TileDivisibilityIntoTiles[level].x = 1 << currentExponent.x;
    full__TileDivisibilityIntoTiles[level].y = 1 << currentExponent.y;

    auto lastFullTileDivisibilityIntoPixels = glm::uvec2(1,1);
    auto lastFullTileDivisibilityIntoTiles  = glm::uvec2(1,1);

    if(level + 1 <= lastLevel){
      lastFullTileDivisibilityIntoPixels = full__TileDivisibilityIntoPixels[level+1];
      lastFullTileDivisibilityIntoTiles  = full__TileDivisibilityIntoTiles [level+1];
    }
    
    full__TileDivisibilityIntoPixels[level] =
      lastFullTileDivisibilityIntoPixels * full__TileDivisibilityIntoTiles[level];

    borderTileDivisibilityIntoPixels[level] = windowSize % full__TileDivisibilityIntoPixels[level];

    borderTileDivisibilityIntoTiles[level] = 
      divRoundUp(borderTileDivisibilityIntoPixels[level],lastFullTileDivisibilityIntoPixels);

    borderTileDivisibilityIntoFullTiles[level] = 
      borderTileDivisibilityIntoPixels[level] / lastFullTileDivisibilityIntoPixels;

    exponentCounter += currentExponent;
    oddLevel = !oddLevel;

  }

  hdtSize.push_back(windowSize);
  for(size_t l=nofLevels-1;l>0;--l)
    hdtSize.push_back(divRoundUp(windowSize,full__TileDivisibilityIntoPixels.at(l)));
  std::reverse(std::begin(hdtSize),std::end(hdtSize));

}
