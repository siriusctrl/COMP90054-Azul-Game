import sys

from model import TileDisplay

sys.path.append("players/StaffTeamEasy")

from advance_model import *
from collections import Counter
from copy import deepcopy


def seeTile(tile_grab: TileGrab):
    print(
        [tile_grab.tile_type,
         tile_grab.number,
         tile_grab.pattern_line_dest,
         tile_grab.num_to_pattern_line,
         tile_grab.num_to_floor_line])


class myPlayer(AdvancePlayer):
    # initialize
    # The following function should not be changed at all
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.weights = Counter()
        # decay for Q value
        self.discount = 0.9
        # learning rate for the Q value
        self.alpha = 0.2
        # exploration and exploitation rule for epsilon greedy
        self.epsilon = 0.8

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state: GameState):
        # TODO ：load the weights from disk at here
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated, 
    # a random action will be selected, and you will receive 
    # a timeout warning
    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        # move[1] is factory ID that illustrate the source of tile, -1 for center

        # print(self.getScore(game_state))
        # for i in moves:
        #     print(i)
        #     print(self.seeTile(i[2]))
        return random.choice(moves)

    def getScore(self, game_state: GameState):
        return game_state.players[self.id].score

    def getQValue(self, game_state: GameState, action) -> float:
        """get the Q value for a specify state with the performed action"""
        # todo : change this part to suit featureExtractor
        q_value = 0.0
        features = self.featureExtractor(game_state, action)
        for key in features.keys():
            q_value += self.weights[key] * features[key]
        return q_value

    def update(self, game_state: GameState, action, next_state: GameState, reward):
        """
            update weight, this will be called at the beginning of each state to
            update the parameters for previous state and action
        """
        pass

    def featureExtractor(self, game_state: GameState, move: (Move, int, TileGrab)) -> dict:
        """
        return the feature that we extract from a specific game state
        that can be used for Q value calculation
        :return a dictionary that contains the value we want to use in this game
        """
        features = []
        # take from center
        if move[1] == -1:
            tile_display: TileDisplay = game_state.centre_pool
            if not game_state.first_player_taken:
                features.append(0)
            else:
                features.append(1)
        # take from factory
        else:
            tile_display: TileDisplay = game_state.factories[move[1]]

        # number of tile that going to take
        features.append(tile_display.total)
        #
        # todo : add more features
        # key is (take from center/factory, target_row)
        return {(move[0], move[-1]): features}

    def getNextState(self, game_state, action) -> GameState:
        """give a state and action, return the next state"""
        next_state: GameState = deepcopy(game_state)
        next_state.ExecuteMove(self.id, action)
        return next_state

    def getReward(self, state: GameState, move: (Move, int, TileGrab)):
        """ return the reward for certain next state """
        pass

    def expectScore(self, state: GameState):
        """ calculate the expected reward for a state, including the end of game score"""
        # todo : finish this and use it as a feature
        pass
