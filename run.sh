#!/usr/bin/env bash
RED=minmax_prune_3
BLUE=naive_player

python runner.py --redName=$RED --blueName=$BLUE -r $RED -b $BLUE -w 1 -m 100 -q