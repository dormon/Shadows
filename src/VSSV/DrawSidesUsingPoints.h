#pragma once

#include <VSSV/DrawSides.h>
#include <geGL/geGL.h>
#include <Vars/Vars.h>

class Adjacency;
class DrawSidesUsingPoints: public DrawSides{
  public:
    DrawSidesUsingPoints(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};
