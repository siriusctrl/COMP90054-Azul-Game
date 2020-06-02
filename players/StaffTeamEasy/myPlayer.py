"""
The University of Melbourne
author: Wenrui Zhang 872427
The implementation of miniMax algorithm
"""

import sys
sys.path.append("players/StaffTeamEasy")

from advance_model import *
from utils import *
from copy import deepcopy


class myPlayer(AdvancePlayer):
    TURN_IGNORE_NEIGHBOUR_BONUS = 3
    TURN_IGNORE_FINAL_BONUS = 3

    def __init__(self, _id):
        super().__init__(_id)
        self.turn = 0

        # weight settings, minimal scale of weight is 0.00001
        self.neighbour_weight = 0.01
        self.final_bonus_weight = 0.4
        self.long_empty_penalty = -0.35
        self.long_fill_bonus = 0.01
        self.fill_bonus = 0.05

        # the parameter for the first move of a turn
        self.is_my_first_move = False
        self.first_move = None

    def StartRound(self, game_state):
        # we have an increasing focus on the bonus as the game goes
        self.turn += 1
        self.final_bonus_weight += 0.2
        self.final_bonus_weight = min(self.final_bonus_weight, 1)

        # check all the moves at the start of the round
        self.is_my_first_move = True
        moves = game_state.players[self.id].GetAvailableMoves(game_state)
        for move in moves:
            # The strategy for round 1
            if self.turn == 1:
                tile = move[2]
                # if it is the blue tile and can add to the middle, perform it
                if tile.tile_type == 0:
                    if tile.number == 3 and tile.pattern_line_dest == 2:
                        return move
        return None

    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        """
        using the algorithm of miniMax to select the move
        the idea is to select some best actions first, then select some worst action to me
        for the opponent based on my selection, and based on his selection, choose
        my best selections and keep repeating until no time left
        The benefit is to lower bound the worst that opponent can lead to
        """

        # First of all, consider the first move decided at the start of the turn
        if self.is_my_first_move:
            self.is_my_first_move = False
            if self.first_move is not None:
                if self.first_move in moves:
                    return self.first_move

        # If there is no best obvious move, take consider of all possible moves
        best_score = float("-inf")
        best_move = None

        # Right now the depthest level it can reach is only max without min due to time
        depth = 0

        for move in moves:
            move_score = self.MiniMax(move, game_state, depth, self.id)
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_move

    def MiniMax(self, move: (Move, int, TileGrab), game_state: GameState, depth: int, player_id: int):
        current_state = deepcopy(game_state)
        expected_score, _ = current_state.players[self.id].ScoreRound()
        # finish diving when it is the highest depth or the end of the game
        if depth == 0:
            return self.ExpectScoreAfterMyMove(move, current_state, expected_score)

        # get the state after the move
        next_state = deepcopy(game_state)
        next_state.ExecuteMove(player_id, move)
        next_player_id = abs(player_id - 1)
        new_depth = depth - 1

        # check all possible moves for next player, if none, return current state value
        next_player_moves = next_state.players[next_player_id].GetAvailableMoves(next_state)
        if len(next_player_moves) == 0:
            return self.ExpectScoreAfterMyMove(move, current_state, expected_score)

        # maximum score part
        if next_player_id == self.id:
            best_score = float("-inf")
            for my_move in next_player_moves:
                move_score = self.MiniMax(my_move, next_state, new_depth, next_player_id)
                best_score = max(best_score, move_score)
            return best_score

        # minimal score part
        else:
            # choose the worst score that my opponent may leads to
            worst_score = float("inf")
            for opponent_move in next_player_moves:
                move_score = self.MiniMax(opponent_move, next_state, new_depth, next_player_id)
                worst_score = min(worst_score, move_score)
            return worst_score

    def ExpectScoreAfterMyMove(self, move: (Move, int, TileGrab), game_state: GameState, neighbour_bonus: float):
        """
        The most significant function that calculate the score for the move
        """
        state_after_move = deepcopy(game_state)
        state_after_move.ExecuteMove(self.id, move)
        my_state = state_after_move.players[self.id]

        # Feature 1: The neighbour bonus
        # currently, if it is the first turn, shrink the positive reward to minimal size
        expected_score, _ = my_state.ScoreRound()
        score_changes = expected_score - neighbour_bonus
        if self.turn <= self.TURN_IGNORE_NEIGHBOUR_BONUS and score_changes > 0:
            final_score = score_changes * self.neighbour_weight
        else:
            final_score = score_changes

        # Feature 2: The column or diagonal or row bonus
        # first turn 0.6, second turn 0.8 then 1
        bonus = my_state.EndOfGameScore()
        if self.turn <= self.TURN_IGNORE_FINAL_BONUS:
            final_score += bonus * self.final_bonus_weight
        else:
            final_score += bonus

        # Feature 3: Become the first player for next turn
        # we may want the first player, according to the strategy, it finishes at turn 5
        if move[0] == Move.TAKE_FROM_CENTRE and not game_state.first_player_taken:
            # first two turn it is not so important, but better if no other choice
            if self.turn < 3:
                final_score += 0.00001
            elif self.turn < 5:
                final_score += 1.00001

        # Feature 4: Penalise when add only a few grad to a long pattern
        tile_grab = move[2]
        target_line = tile_grab.pattern_line_dest
        num_add_to_line = tile_grab.num_to_pattern_line
        line_n = game_state.players[self.id].lines_number
        # when it is not add to the floor
        if tile_grab.pattern_line_dest != -1:
            # when the row currently is empty
            if line_n[target_line] == 0:
                # total capacity - # we going to add
                final_score += self.long_empty_penalty * (target_line + 1 - num_add_to_line)

        # Feature 5: give a slightly higher point to collect more
        final_score += num_add_to_line * self.fill_bonus

        # Feature 6: When the added tile is close the center, it is better
        tile_colour = tile_grab.tile_type
        if target_line == 0 or target_line == 4:
            if tile_colour == 1 or tile_colour == 3:
                final_score += 0.00001
            elif tile_colour == 2:
                final_score += 0.00002
        elif target_line == 1 or target_line == 3:
            if tile_colour == 0 or tile_colour == 4:
                final_score += 0.00001
            if tile_colour == 1 or tile_colour == 3:
                final_score += 0.00002
            elif tile_colour == 2:
                final_score += 0.00003
        elif target_line == 2:
            if tile_colour == 0 or tile_colour == 4:
                final_score += 0.00002
            if tile_colour == 1 or tile_colour == 3:
                final_score += 0.00003
            elif tile_colour == 2:
                final_score += 0.00004

        # The following features are not very useful, but I leave it here for checking
        # Feature 7: If there are some same tiles with same amount in the factory or in the center
        # we don't need to fill the current tile so hurry
        # However, it should be implemented at higher level, not here!!!
        # find_same = 0
        # for i in [0, 5]:
        #     if game_state.factories[0].tiles[tile_colour] == num_add_to_line:
        #         find_same += 1
        # if find_same < 3:
        #     if game_state.centre_pool.tiles[tile_colour] == num_add_to_line:
        #         find_same += 1
        #
        # if find_same >= 3:
        #     final_score -= 10

        # # Feature 8: give bonus to the pattern line that already have some tile
        # if tile_grab.pattern_line_dest != -1:
        #     final_score += line_n[tile_grab.pattern_line_dest] * 0.0001

        return final_score


