#!/usr/bin/env bash
FILE=StaffTeamEasy.myPlayer

python runner.py --redName=$FILE --blueName=minmax_prune -r $FILE -b minmax_prune_3 -w 2 -m 100 -q