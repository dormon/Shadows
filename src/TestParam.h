#pragma once

#include<iostream>
#include<ArgumentViewer/ArgumentViewer.h>

struct TestParam{
  std::string name                 = "";
  std::string flyKeyFileName       = "";
  size_t      flyLength            = 0;
  size_t      framesPerMeasurement = 5;
  std::string outputName           = "measurement";
  TestParam(){}
  TestParam(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
};
