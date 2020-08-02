#!/bin/sh
./build/linux2/Shadows --model /media/devel/models/2tri/2tri.obj --light 0 5 0 1 --camera-speed 0.1 --camera-remember --method sintorn2 --window-size 1024 1024 sintorn2Param \{ ffc 0 morePlanes 0 \} "$@"
