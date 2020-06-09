#!/bin/sh
./build/linux2/Shadows --model /media/devel/models/sponza/sponza.obj --camera-speed 5 --method rssv --window-size 1024 1024 rssvParam \{ memoryOptim 1 scaledQuantization 1 \} "$@"
