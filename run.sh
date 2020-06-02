#!/usr/bin/env bash
RED=minmax_prune_3
BLUE=StaffTeamEasy.myPlayer

python runner.py --redName=$RED --blueName=$BLUE -r $RED -b $BLUE -w 2 -m 10 -q -p