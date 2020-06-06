#!/bin/sh
./build/linux2/Shadows --model /media/devel/models/2tri/2tri.obj --camera-speed 0.1 --camera-remember --method rssv --window-size 1024 1024 rssvParam \{ memoryOptim 1 scaledQuantization 1 \} "$@"
