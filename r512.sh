#!/bin/sh
./build/linux2/Shadows --model /media/devel/models/2tri/2tri.obj --light 0 5 0 1 --camera-speed 0.1 --method rssv --window-size 512 512 rssvParam \{ memoryOptim 1 scaledQuantization 1 minZBits 6 \} "$@"
