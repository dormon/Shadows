#!/bin/sh
BASEPARAM="--model /media/devel/models/sponza/sponza.obj --camera-speed 5 --method rssv --window-size 1024 1024"
./build/linux2/Shadows ${BASEPARAM} rssvParam \{ memoryOptim 1 scaledQuantization 1 traverseSilhouettes 0 traverseTriangles 1 \} "$@"
