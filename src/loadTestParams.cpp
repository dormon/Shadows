#include <loadTestParams.h>

void loadTestParams(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&arg){
  vars.addString("test.name"                ) = arg->gets  ("--test"                     ,""           ,"name of test - fly or empty"                                    );
  vars.addString("test.flyKeyFileName"      ) = arg->gets  ("--test-fly-keys"            ,""           ,"filename containing fly keyframes - csv x,y,z,vx,vy,vz,ux,uy,uz");
  vars.addSizeT ("test.flyLength"           ) = arg->getu32("--test-fly-length"          ,1000         ,"number of measurements, 1000"                                   );
  vars.addSizeT ("test.framesPerMeasurement") = arg->getu32("--test-framesPerMeasurement",5            ,"number of frames that is averaged per one measurement point"    );
  vars.addString("test.outputName"          ) = arg->gets  ("--test-output"              ,"measurement","name of output file"                                            );

}
