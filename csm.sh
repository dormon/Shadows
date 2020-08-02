#!/bin/sh
DRI_PRIME=1 ./build/linux2/Shadows --model /media/devel/models/sponza/sponza.obj --method cubeShadowMapping --verbose --window-size 1024 1024 --shadowMap-faces 1 "$@"
