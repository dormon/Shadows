#!/bin/sh
BASEPARAM="--model /media/devel/models/2tri/2tri.obj --light 0 5 0 1 --camera-speed 0.1 --camera-remember --method rssv --window-size 1024 1024"
./build/linux2/Shadows ${BASEPARAM}  rssvParam \{ memoryOptim 1 scaledQuantization 1 traverseSilhouettes 0 traverseTriangles 1 \} "$@"
