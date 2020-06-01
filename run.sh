file=naive_player

#python runner.py -r $file -b StaffTeamEasy.myPlayer \
#    --delay 0 --blueName=$file --redName=myPlayer -m 1 --setRandomSeed=50 -q


python runner.py -r $file -b StaffTeamEasy.myPlayer \
    --delay 0 --redName=$file --blueName=myPlayer -m 1 -q
