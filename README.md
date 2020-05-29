<!-- # AZUL
This repository contains a framework to support policy learning for the boardgame AZUL, published by Plan B Games. The purpose of this framework is to allow students to implement algorithms for learning AI players for the game and evaluate the performance of these players against human/other AI players. 

Students making use of the framework will need to create a Player subclass for their AI player that selects moves on the basis of a learned policy, and write code to learn their policy on the basis of repeated simulations of the game.

Some information about the game:
- https://en.wikipedia.org/wiki/Azul_(board_game)
- https://www.ultraboardgames.com/azul/game-rules.php
- https://boardgamegeek.com/boardgame/230802/azul
- https://www.planbgames.com/en/news/azul-c16.html -->

# AZUL GUI

This repository contains a framework to support developing autonomous agents for the boardgame AZUL, published by Plan B Games. The game frame is forked from [Michelle Blom](https://github.com/michelleblom)'s repository, and GUI is developed by [Guang Hu](https://github.com/guanghuhappysf128) and  [Ruihan Zhang](https://github.com/zhangrh93). The code is in Python 3.

Students should be able to use this frame and develop their own agent under the directory players. This framework is able to run AZUL with two agents in a 1v1 game, and GUI will allow user to navigate through recorded game states. In addition, a replay can be saved and played for debug purpose.

Some information about the game:
- https://en.wikipedia.org/wiki/Azul_(board_game)
- https://boardgamegeek.com/boardgame/230802/azul
- https://www.planbgames.com/en/news/azul-c16.html
- https://github.com/michelleblom/AZUL

# Setting up the environment

Python 3 is required, and library tkinter should be installed along with python 3.

The code uses three library that required to be installed: ```numpy```,```func_timeout```,```tqdm```, which can be done with the following command:
```bash
pip install numpy tqdm func_timeout
```
If have both python 2 and python 3 installed, you might need to use following command:
```bash
pip3 install numpy tqdm func_timeout
```

In some OS, such as Ubuntu 18.04, tk is not pre-installed, you might get a mistake such as:
```
ModuleNotFoundError: No module named 'tkinter'
```
It can be fixed with:
```bash
sudo apt-get install python3-tk
```

# How to run it?

The code example can be run with command:
```bash
python runner.py
```
, which will run the game with two default players (naive_player). 

A standard example to run the code would be:
```bash
python runner.py -r naive_player -b StaffTeamEasy.myPlayer --blueName='872427' -q -m 10
```

Other command line option can be viewed with argument: ```-h``` or ```--help```
* Options:
*   -h, --help            show this help message and exit
*   -r RED, --red=RED     Red team player file (default: naive_player)
*   -b BLUE, --blue=BLUE  Blue team player file (default: naive_player)
*   --redName=REDNAME     Red team name (default: Red NaivePlayer)
*   --blueName=BLUENAME   Blue team name (default: Blue NaivePlayer)
*   -t, --textgraphics    Display output as text only (default: False)
*   -q, --quiet           No text nor graphics output, only show game info
*   -Q, --superQuiet      No output at all
*   -w WARNINGTIMELIMIT, --warningTimeLimit=WARNINGTIMELIMIT Time limit for a warning of one move in seconds (default: 1)
*   --startRoundWarningTimeLimit=STARTROUNDWARNINGTIMELIMIT Time limit for a warning of initialization for each round in seconds (default: 5)
*   -n NUMOFWARNINGS, --numOfWarnings=NUMOFWARNINGS Num of warnings a team can get before fail (default: 3)
*   -m MULTIPLEGAMES, --multipleGames=MULTIPLEGAMES Run multiple games in a roll
*   --setRandomSeed=SETRANDOMSEED Set the random seed, otherwise it will be completely random (default: 90054)
*   -s, --saveGameRecord  Writes game histories to a file (named by teams' names and the time they were played) (default: False)
*   -o OUTPUT, --output=OUTPUT output directory for replay and log (default: output)
*   -l, --saveLog         Writes player printed information into a log file(named by the time they were played)
*   --replay=REPLAY       Replays a recorded game file by a relative path
*   --delay=DELAY         Delay action in a play or replay by input (float) seconds (default 0.1)



When a game ends, the GUI will pause to allow user selecting each states on listbox (right side of the window), and it will change accordingly. And replay file will be generated once the GUI window is closed.



***For Debug purpose:***
***Please use the Example.ipynb to start***

**Extra function**
- timeout limit
- timeout warning and fail
- replay system
- GUI displayer (allow switch)
- delay time setting

**class and parameters**

*AdvancedRunner*

Runner with timelimit, timeout warnings, displayer, replay system. It returns a replay file.

*ReplayRunner*

Use replay file to unfold a replay

*GUIGameDisplayer*

GUI game displayer, you coud click items in the list box and use arrow keys to select move.
