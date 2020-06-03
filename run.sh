#!/usr/bin/env bash
RED=naive_player
BLUE=StaffTeamEasy.myPlayer

python runner.py --redName=$RED --blueName=$BLUE -r $RED -b $BLUE -w 2 -m 100 -q -p