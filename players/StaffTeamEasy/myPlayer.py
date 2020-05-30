
import sys
sys.path.append("players/StaffTeamEasy")

from advance_model import *
from utils import *
from copy import deepcopy
import random

def seeTile(tile_grab: TileGrab):
    print(
        [tile_grab.tile_type,
         tile_grab.number,
         tile_grab.pattern_line_dest,
         tile_grab.num_to_pattern_line,
         tile_grab.num_to_floor_line])


class myPlayer(AdvancePlayer):
    """
    author: Wenrui Zhang 872427
    The implementation of miniMax algorithm
    """

    # initialize
    # You can add your own data initilazation here, just make sure nothing breaks
    def __init__(self, _id):
        super().__init__(_id)
        self.turn = 0
        self.bonus_weight = 0.25
        self.long_empty_weight = 0.4

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state):
        # we have an increasing focus on the bonus as the game goes
        self.turn += 1
        self.bonus_weight += self.turn * 0.05
        self.bonus_weight = min(self.bonus_weight, 1)
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated,
    # a random action will be selected, and you will receive
    # a timeout warning
    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        """
        using the algorithm of miniMax to select the move
        the idea is to select some best actions first, then select some worst action to me
        for the opponent based on my selection, and based on his selection, choose
        my best selections and keep repeating until no time left
        The benifit is to lower bound the worst that opponent can lead to
        """
        best_score = float("-inf")
        best_move = None
        depth = 0
        # if len(moves) < 15:
        #     depth = 1
        # if len(moves) < 5:
        #     depth = 3

        for move in moves:
            # penalise add only a few grad to a long pattern
            tile_grab: TileGrab = move[-1]
            line_n = game_state.players[self.id].lines_number
            # total capacity - tile already have - # we going to add
            remains = (tile_grab.pattern_line_dest + 1) - line_n[tile_grab.pattern_line_dest] \
                      - tile_grab.num_to_pattern_line

            next_state = deepcopy(game_state)
            next_state.ExecuteMove(self.id, move)
            move_score = self.MiniMax(next_state, depth, abs(self.id - 1)) - remains * self.long_empty_weight
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_move

    def MiniMax(self, game_state: GameState, depth: int, id: int):
        # finish diving when it is the highest depth or the end of the game
        if depth == 0:
            return self.ExpectScoreAfterMyMove(game_state)

        next_player_id = abs(id - 1)
        # maximum score part
        if id == self.id:
            my_moves = game_state.players[next_player_id].GetAvailableMoves(game_state)
            best_score = float("-inf")
            for my_move in my_moves:
                # penalise add only a few grad to a long pattern
                tile_grab: TileGrab = my_move[-1]
                line_n = game_state.players[self.id].lines_number
                # total capacity - tile already have - # we going to add
                remains = (tile_grab.pattern_line_dest + 1) - line_n[tile_grab.pattern_line_dest] \
                          - tile_grab.num_to_pattern_line

                next_state = deepcopy(game_state)
                next_state.ExecuteMove(next_player_id, my_move)
                move_score = self.MiniMax(next_state, depth - 1, next_player_id) - remains * self.long_empty_weight
                best_score = max(best_score, move_score)
            if best_score == float("-inf"):
                return self.ExpectScoreAfterMyMove(game_state)
            return best_score

        # minimal score part
        else:
            opponent_moves = game_state.players[next_player_id].GetAvailableMoves(game_state)
            worst_score = float("inf")
            for opponent_move in opponent_moves:
                next_state = deepcopy(game_state)
                next_state.ExecuteMove(next_player_id, opponent_move)
                move_score = self.MiniMax(next_state, depth - 1, next_player_id)
                worst_score = min(worst_score, move_score)
            if worst_score == float("inf"):
                return self.ExpectScoreAfterMyMove(game_state)
            return worst_score

    def ExpectScoreAfterMyMove(self, state: GameState):
        # The standard score achieve based on the
        my_state: PlayerState = state.players[self.id]
        expected_score, _ = my_state.ScoreRound()
        bonus = my_state.EndOfGameScore()

        return expected_score + bonus * self.bonus_weight







    def ExpectScoreAfterOppoMove(self, state: GameState):
        my_state: PlayerState = state.players[self.id]
        expected_score, _ = my_state.ScoreRound()
        bonus = my_state.EndOfGameScore()

        return expected_score + bonus * self.bonus_weight

