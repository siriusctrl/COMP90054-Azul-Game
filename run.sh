#!/usr/bin/env bash
RED=naive_player
BLUE=compare.myPlayer\(1\)

python runner.py --redName=$RED --blueName=$BLUE -r $RED -b $BLUE -w 2 -m 100 -q -p