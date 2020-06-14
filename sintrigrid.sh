#!/bin/sh
BASEPARAM="--model /media/devel/models/trigrid/trigrid.obj --light 0 5 0 1 --camera-speed 0.01 --camera-remember --method sintorn2 --window-size 1024 1024"
TRI=" traverseTriangles 1 computeLastLevelTriangles 0 exactTriangleAABB 1 computeTriangleBridges 1 "
./build/linux2/Shadows ${BASEPARAM}  rssvParam \{ memoryOptim 1 scaledQuantization 1 traverseSilhouettes 0 ${TRI} \} "$@"
