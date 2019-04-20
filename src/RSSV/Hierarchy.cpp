#include <algorithm>
#include <iostream>

#include <RSSV/Hierarchy.h>

using namespace rssv;
using namespace glm;
using namespace std;

template<typename T>
T divRoundUp(T a,T b){
  return (a/b) + static_cast<T>((a%b)>0);
}

uvec2 divRoundUp(uvec2 const&a,uvec2 const&b){
  return uvec2(divRoundUp(a.x,b.x),divRoundUp(a.y,b.y));
}

uint32_t ilog2(uint32_t a){
  return static_cast<uint32_t>(glm::log2(static_cast<float>(a)));
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
  uvec2 fullTileCount                         ;
  uvec2 borderTileSize                        ;
  uvec2 borderTileSizeInPixels                ;
  vec2  fullTileSizeInClipSpace               ;
  vec2  borderTileSizeInClipSpace             ;

  auto const computeLevel = [&](){
    tileCount                  = divRoundUp(levelSize,fullTileSize);
    fullTileSizeInPixels      *= fullTileSize;
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


  reverse(this->levelSize                );
  reverse(this->tileCount                );
  reverse(this->fullTileSize             );
  reverse(this->fullTileCount            );
  reverse(this->borderTileSize           );
  reverse(this->fullTileSizeInPixels     );
  reverse(this->fullTileSizeInClipSpace  );
  reverse(this->borderTileSizeInPixels   );
  reverse(this->borderTileSizeInClipSpace);
}

void rssv::printHierarchy(Hierarchy const&h){
  std::cerr << "level size: " << std::endl;
  for(auto const&x:h.levelSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "tile count: " << std::endl;
  for(auto const&x:h.tileCount)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size: " << std::endl;
  for(auto const&x:h.fullTileSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size in pixels: " << std::endl;
  for(auto const&x:h.fullTileSizeInPixels)
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

