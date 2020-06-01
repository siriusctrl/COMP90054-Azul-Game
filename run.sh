#!/usr/bin/env bash
RED=minmax_prune
BLUE=StaffTeamEasy.myPlayer

python runner.py --redName=$RED --blueName=$BLUE -r $RED -b $BLUE -w 1 -m 15 -q