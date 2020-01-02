#include <algorithm>
#include <iostream>

#include <RSSV/Hierarchy.h>
#include <divRoundUp.h>

using namespace rssv;
using namespace glm;
using namespace std;

uint32_t ilog2(uint32_t a){
  return static_cast<uint32_t>(glm::log2(static_cast<float>(a)));
}

uvec2 ilog2(uvec2 const&a){
  return uvec2(ilog2(a.x),ilog2(a.y));
}

template<typename T>
void reverse(std::vector<T>&v){
  std::reverse(v.begin(),v.end());
}


rssv::Hierarchy::Hierarchy(glm::uvec2 const&windowSize,uint32_t branchingFactor){
  glm::uvec2 fullTileSize;
  fullTileSize.y = 1<<(ilog2(branchingFactor)/2);
  fullTileSize.x = branchingFactor / fullTileSize.y;

  uvec2 levelSize                 = windowSize;
  uvec2 tileCount                             ;
  uvec2 fullTileSizeInPixels      = uvec2(1)  ;
  uvec2 fullTileExponent                      ;
  uvec2 fullTileCount                         ;
  uvec2 borderTileSize                        ;
  uvec2 borderTileSizeInPixels                ;
  vec2  fullTileSizeInClipSpace               ;
  vec2  borderTileSizeInClipSpace             ;

  auto const computeLevel = [&](){
    tileCount                  = divRoundUp(levelSize,fullTileSize);
    fullTileSizeInPixels      *= fullTileSize;
    fullTileExponent           = ilog2(fullTileSizeInPixels);
    fullTileCount              = windowSize / fullTileSizeInPixels;
    borderTileSize             = levelSize-fullTileCount*fullTileSize;
    borderTileSizeInPixels     = windowSize-fullTileCount*fullTileSizeInPixels;
    fullTileSizeInClipSpace    = vec2(fullTileSizeInPixels) / vec2(windowSize) * 2.f;
    borderTileSizeInClipSpace  = vec2(borderTileSizeInPixels) / vec2(windowSize) * 2.f;
    this->levelSize                .push_back(levelSize                );
    this->tileCount                .push_back(tileCount                );
    this->fullTileSize             .push_back(fullTileSize             );
    this->fullTileCount            .push_back(fullTileCount            );
    this->borderTileSize           .push_back(borderTileSize           );
    this->fullTileSizeInPixels     .push_back(fullTileSizeInPixels     );
    this->fullTileExponent         .push_back(fullTileExponent         );
    this->borderTileSizeInPixels   .push_back(borderTileSizeInPixels   );
    this->fullTileSizeInClipSpace  .push_back(fullTileSizeInClipSpace  );
    this->borderTileSizeInClipSpace.push_back(borderTileSizeInClipSpace);
  };

  while(levelSize.x > 1 || levelSize.y > 1){
    bool xDone = levelSize.x == 1;
    bool yDone = levelSize.y == 1;
    while(fullTileSize.x >= levelSize.x*2 && !yDone){
      fullTileSize.x/=2;
      fullTileSize.y*=2;
    }
    while(fullTileSize.y >= levelSize.y*2 && !xDone){
      fullTileSize.y/=2;
      fullTileSize.x*=2;
    }
    computeLevel();
    levelSize    = divRoundUp(levelSize,fullTileSize);
    fullTileSize = uvec2(fullTileSize.y,fullTileSize.x);
  }
  computeLevel();

  this->levelSize                .pop_back();
  this->tileCount                .pop_back();
  this->fullTileSize             .pop_back();
  this->fullTileCount            .pop_back();
  this->borderTileSize           .pop_back();
  this->fullTileSizeInPixels     .pop_back();
  this->fullTileExponent         .pop_back();
  this->fullTileSizeInClipSpace  .pop_back();
  this->borderTileSizeInPixels   .pop_back();
  this->borderTileSizeInClipSpace.pop_back();

  reverse(this->levelSize                );
  reverse(this->tileCount                );
  reverse(this->fullTileSize             );
  reverse(this->fullTileCount            );
  reverse(this->borderTileSize           );
  reverse(this->fullTileSizeInPixels     );
  reverse(this->fullTileExponent         );
  reverse(this->fullTileSizeInClipSpace  );
  reverse(this->borderTileSizeInPixels   );
  reverse(this->borderTileSizeInClipSpace);

  this->fullTileExponentPrev = this->fullTileExponent;
  for(auto&x:this->fullTileExponentPrev)
    x -= this->fullTileExponentPrev.back();

  for(auto const&x:this->fullTileExponent)
    this->fullTileMask.push_back(uvec2((1u<<x)-1u));

  for(auto const&x:this->fullTileExponentPrev)
    this->fullTileMaskPrev.push_back(uvec2((1u<<x)-1u));

  this->nofLevels = (uint32_t)this->levelSize.size();
}

void rssv::printHierarchy(Hierarchy const&h){
  std::cerr << "level size / active threads: " << std::endl;
  for(auto const&x:h.levelSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "tile count / glDispatchCompute: " << std::endl;
  for(auto const&x:h.tileCount)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size / workgroup size: " << std::endl;
  for(auto const&x:h.fullTileSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size in pixels: " << std::endl;
  for(auto const&x:h.fullTileSizeInPixels)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;
  
  std::cerr << "full tile exponent: " << std::endl;
  for(auto const&x:h.fullTileExponent)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile mask: " << std::endl;
  for(auto const&x:h.fullTileMask)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile exponent prev: " << std::endl;
  for(auto const&x:h.fullTileExponentPrev)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile mask prev: " << std::endl;
  for(auto const&x:h.fullTileMaskPrev)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full size in clip space: " << std::endl;
  for(auto const&x:h.fullTileSizeInClipSpace)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile count: " << std::endl;
  for(auto const&x:h.fullTileCount)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size: " << std::endl;
  for(auto const&x:h.borderTileSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size in pixels: " << std::endl;
  for(auto const&x:h.borderTileSizeInPixels)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size in clip space: " << std::endl;
  for(auto const&x:h.borderTileSizeInClipSpace)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;


}

