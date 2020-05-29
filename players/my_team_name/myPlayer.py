# This file will be used in the competition
# Please make sure the following functions are well defined

from advance_model import *
from utils import *
# if you want to use random, you have to import it here
import random

# if you want to import other py file in the same directory as your myPlayer.py
import sys
sys.path.append("players/my_team_name/")

class myPlayer(AdvancePlayer):

    # initialize
    # You can add your own data initilazation here, just make sure nothing breaks
    def __init__(self,_id):
        super().__init__(_id)

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and 
    # you will receive a timeout warning
    def StartRound(self,game_state):
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated, 
    # a random action will be selected, and you will receive 
    # a timeout warning
    def SelectMove(self, moves, game_state):
        return random.choice(moves)
