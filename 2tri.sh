#!/bin/sh
BASEPARAM="--model /media/devel/models/2tri/2tri.obj --light 0 5 0 1 --camera-speed 0.1 --camera-remember --method rssv --window-size 1024 1024"
TRI=" traverseTriangles   1 computeLastLevelTriangles   1 exactTriangleAABB   1 computeTriangleBridges   1 "
SIL=" traverseSilhouettes 1 computeLastLevelSilhouettes 1 exactSilhouetteAABB 1 computeSilhouetteBridges 1 "
./build/linux2/Shadows ${BASEPARAM}  rssvParam \{ memoryOptim 1 scaledQuantization 1 persistentWG 128 performMerge 0 bias 0 ${SIL} ${TRI} \} "$@"
