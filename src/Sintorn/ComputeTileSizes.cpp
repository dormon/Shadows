#include <Sintorn/ComputeTileSizes.h>
#include <Sintorn/Tiles.h>

#include <iostream>
#include <FunctionPrologue.h>
#include <Barrier.h>
#include <glm/glm.hpp>
#include <Vars/Caller.h>

void computeTileDivisibility(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","wavefrontSize","windowSize");

  auto const windowSize = *vars.get<glm::uvec2>("windowSize");
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");

  auto&tileDivisibility = vars.reCreateVector<glm::uvec2>("sintorn.tileDivisibility");

  chooseTileSizes(tileDivisibility,windowSize,wavefrontSize);
  
  auto nofLevels = tileDivisibility.size();
  for(size_t l=0;l<nofLevels;++l)
    std::cerr<<"TileDivisibility: "<<tileDivisibility[l].x<<" "<<tileDivisibility[l].y<<std::endl;
}

void computeTileSizeInPixel(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","sintorn.tileDivisibility");

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels = tileDivisibility.size();

  auto&tileSizeInPixels = vars.reCreateVector<glm::uvec2>("sintorn.tileSizeInPixels");
  //compute tile size in pixels
  tileSizeInPixels.resize(nofLevels,glm::uvec2(1u,1u));
  for(size_t l=0;l<nofLevels;++l)
    for(size_t m=l+1;m<nofLevels;++m)
      tileSizeInPixels[l]*=tileDivisibility[m];
  
  for(unsigned l=0;l<nofLevels;++l)
    std::cerr<<"TileSizeInPixels: "<<tileSizeInPixels[l].x<<" "<<tileSizeInPixels[l].y<<std::endl;
}

void computeTileCount(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","sintorn.tileDivisibility");

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels = tileDivisibility.size();

  auto&tileCount = vars.reCreateVector<glm::uvec2>("sintorn.tileCount");
  //compute level size
  tileCount.resize(nofLevels,glm::uvec2(1u,1u));
  std::cerr << "_tileCount.size: " << tileCount.size() << std::endl;
  std::cerr << "nofLevels: " << nofLevels << std::endl;
  for(size_t l=0;l<nofLevels;++l)
    for(size_t m=l;m<nofLevels;++m){
      auto const&td = tileDivisibility.at(l); 
      auto& tc = tileCount.at(m);
      tc *= td;
    }

  for(size_t l=0;l<nofLevels;++l)
    std::cerr<<"TileCount: "<<tileCount[l].x<<" "<<tileCount[l].y<<std::endl;
}


void computeTileSizeInClipSpace(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","sintorn.tileSizeInPixels","windowSize");

  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const windowSize       = *vars.get<glm::uvec2>("windowSize");

  auto&tileSizeInClipSpace = vars.reCreateVector<glm::vec2>("sintorn.tileSizeInClipSpace");

  //compute tiles sizes in clip space
  for(auto const&x:tileSizeInPixels)
    tileSizeInClipSpace.push_back(glm::vec2(2.f)/glm::vec2(windowSize)*glm::vec2(x));
  
  auto nofLevels = tileSizeInPixels.size();
  for(size_t l=0;l<nofLevels;++l)
    std::cerr<<"TileSizeInClip: "<<tileSizeInClipSpace[l].x<<" "<<tileSizeInClipSpace[l].y<<std::endl;
}

void computeUsedTiles(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn","sintorn.tileDivisibility","windowSize");

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const windowSize       = *vars.get<glm::uvec2>("windowSize");

  auto&usedTiles = vars.reCreateVector<glm::uvec2>("sintorn.usedTiles");

  auto divRoundUp = [](uint32_t x,uint32_t y)->uint32_t{return (x/y)+((x%y)?1:0);};
  usedTiles.resize(nofLevels,glm::uvec2(0u,0u));
  usedTiles.back() = windowSize;
  for(int l=(int)nofLevels-2;l>=0;--l){
    usedTiles[l].x = divRoundUp(usedTiles[l+1].x,tileDivisibility[l+1].x);
    usedTiles[l].y = divRoundUp(usedTiles[l+1].y,tileDivisibility[l+1].y);
  }

  for(size_t l=0;l<nofLevels;++l)
    std::cerr<<"UsedTiles: "<<usedTiles[l].x<<" "<<usedTiles[l].y<<std::endl;
}


void computeTileSizes(vars::Vars&vars){
  computeTileDivisibility(vars);
  computeTileSizeInPixel(vars);
  computeTileSizeInClipSpace(vars);
  computeTileCount(vars);
  computeUsedTiles(vars);
}
