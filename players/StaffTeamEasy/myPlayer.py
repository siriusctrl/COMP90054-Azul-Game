import sys
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
        self.weights = {"complete": [1], "non-complete": [1]}
        # decay for Q value
        self.discount = 0.9
        # learning rate for the Q value
        self.alpha = 0.2
        # exploration and exploitation rule for epsilon greedy
        self.epsilon = 0.8
        self.other_available = None

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state: GameState):
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated, 
    # a random action will be selected, and you will receive 
    # a timeout warning
    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        # move[1] is factory ID that illustrate the source of tile, -1 for center
        move_collection = dict()

        # # FIXME this will timeout, think another way to consider opponent action
        # for p in game_state.players:
        #     if p.id != self.id:
        #         self.other_available = p.GetAvailableMoves()

        for m in moves:
            move_collection[m] = self.getQValue(game_state, m)

        # find the action with highest Q value
        maxQ = float("-inf")
        curr_max = None
        for key in move_collection.keys():
            if move_collection[key] > maxQ:
                curr_max = key
                maxQ = move_collection[key]

        # use epsilon greedy for now
        # print(maxQ)
        if self.flipCoin():
            return curr_max
        else:
            return random.choice(moves)

    def getQValue(self, game_state: GameState, action) -> float:
        """get the Q value for a specify state with the performed action"""
        q_value = 0.0
        features = self.featureExtractor(game_state, action)

        if "complete" in features:
            key = "complete"
        else:
            key = "non-complete"

        for i in range(len(self.weights[key])):
            q_value += self.weights[key][i] * features[key][i]

        return q_value

    def update(self, game_state: GameState, action) -> None:
        """
            update weight, this will be called at the beginning of each state to
            update the parameters for previous state and action
        """
        next_state = self.getNextState(game_state, action)
        reward = 0

    def featureExtractor(self, game_state: GameState, move: (Move, int, TileGrab)) -> dict:
        """
        return the feature that we extract from a specific game state
        that can be used for Q value calculation

        feature = [expected score, bonus at end]

        :return a dictionary that contains the value we want to use in this game
        """
        features = []
        next_state = self.getNextState(game_state, move)

        if next_state.players[self.id].GetCompletedRows() > 0:
            key = "complete"
        else:
            key = "non-complete"

        # expected score for the current action exec
        features.append(self.expectGain(game_state, next_state))

        # max_score = 0
        # for p in game_state.players:
        #     if p.id != self.id:
        #         max_score = max(p.ScoreRound()[0], max_score)
        #
        # # score difference at the new state - score difference at the previous state
        # features.append((game_state.players[self.id].score - max_score) -
        #                 (next_state.players[self.id].score - max_score))
        #
        # if features[0] > 0.0:
        #     print(features)

        # TODO : add more features here

        return {key: features}

    def getNextState(self, game_state: GameState, action) -> GameState:
        """give a state and action, return the next state"""
        next_state: GameState = deepcopy(game_state)
        next_state.ExecuteMove(self.id, action)
        return next_state

    def expectGain(self, curr_state, next_state):
        curr_expected_score, curr_bonus = self.expectScore(curr_state)
        next_expected_score, next_bonus = self.expectScore(next_state)

        return next_expected_score + next_bonus - curr_expected_score - curr_bonus

    def expectScore(self, state: GameState):
        """
            calculate the expected reward for a state, including the end of game score
            :param state should be deep copied state and applied the selected action
        """
        my_state: PlayerState = state.players[self.id]
        expected_score, _ = my_state.ScoreRound()
        bonus = my_state.EndOfGameScore()
        return expected_score, bonus

    def flipCoin(self) -> bool:
        return True if random.random() < self.discount else False
