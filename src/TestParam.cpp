#include <TestParam.h>

TestParam::TestParam(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  name           = arg->gets("--test", "", "name of test - fly or empty");
  flyKeyFileName = arg->gets(
  "--test-fly-keys", "",
  "filename containing fly keyframes - csv x,y,z,vx,vy,vz,ux,uy,uz");
  flyLength =
  arg->geti32("--test-fly-length", 1000, "number of measurements, 1000");
  framesPerMeasurement = arg->geti32(
  "--test-framesPerMeasurement", 5,
  "number of frames that is averaged per one measurement point");
  outputName =
      arg->gets("--test-output", "measurement", "name of output file");
}
