"""
The University of Melbourne
author: Wenrui Zhang 872427
The implementation of miniMax algorithm
"""
import heapq
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

        self.opponent_id = abs(1 - self.id)
        self.opponent_agent = GreedyAgent(self.opponent_id)
        self.greedy_agent = GreedyAgent(self.id)

    def advance_get_available_actions(self, moves: [(Move, int, TileGrab)], game_state: GameState, player_id: int):
        player_state = game_state.players[player_id]

        result = [[] for _ in range(5)]

        for action in moves:
            move, factory, tile_grab = action

            pattern_tile_numbers = tile_grab.pattern_line_dest + 1

            current_pattern_line_has = player_state.lines_number[tile_grab.pattern_line_dest]

            take_from_center = move == Move.TAKE_FROM_CENTRE

            first_player_taken = take_from_center and not game_state.first_player_taken

            no_tile_to_floor_line = tile_grab.num_to_floor_line == 0
            only_take_first_player_token = tile_grab.num_to_pattern_line == 1 and first_player_taken

            # print(move, factory, seeTile(tile_grab))
            # print(no_tile_to_floor_line, only_take_first_player_token,
            #       (pattern_tile_numbers == current_pattern_line_has + tile_grab.num_to_pattern_line, pattern_tile_numbers, current_pattern_line_has, tile_grab.num_to_pattern_line)
            #       )
            # 1. fill a line and with no tile to floor line or only one 1st player token
            if (no_tile_to_floor_line or only_take_first_player_token) and \
                    pattern_tile_numbers == current_pattern_line_has + tile_grab.num_to_pattern_line and \
                    tile_grab.num_to_pattern_line > 0:
                # print(0)
                result[0].append(action)
                continue

            # 2. not fill a line and with no tile to floor line or only one 1st player token
            if (no_tile_to_floor_line or only_take_first_player_token) and \
                    pattern_tile_numbers > current_pattern_line_has + tile_grab.num_to_pattern_line and \
                    tile_grab.num_to_pattern_line > 0:
                result[1].append(action)
                # print(1)
                continue

            # 3. fill a line and with some tile to floor line or only one 1st player token
            if ((not no_tile_to_floor_line) or only_take_first_player_token) and \
                    pattern_tile_numbers == current_pattern_line_has + tile_grab.num_to_pattern_line and \
                    tile_grab.num_to_pattern_line > 0:
                result[2].append(action)
                # print(2)
                continue

            # 4. not fill a line and with some tile to floor line or only one 1st player token
            if ((not no_tile_to_floor_line) or only_take_first_player_token) and \
                    pattern_tile_numbers > current_pattern_line_has + tile_grab.num_to_pattern_line and \
                    tile_grab.num_to_pattern_line > 0:
                result[3].append(action)
                # print(3)
                continue

            # 5. others
            # print(4)
            result[4].append(action)

        return result

    def StartRound(self, game_state):
        print("------------------ round start ------------------")

        # we have an increasing focus on the bonus as the game goes
        self.turn += 1
        self.final_bonus_weight += 0.2
        self.final_bonus_weight = min(self.final_bonus_weight, 1)

        # check all the moves at the start of the round
        # self.is_my_first_move = True
        # moves = game_state.players[self.id].GetAvailableMoves(game_state)
        # for move in moves:
        #     # The strategy for round 1
        #     if self.turn == 1:
        #         tile = move[2]
        #         # if it is the blue tile and can add to the middle, perform it
        #         if tile.tile_type == 0:
        #             if tile.number == 3 and tile.pattern_line_dest == 2:
        #                 return move
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
        # if self.is_my_first_move:
        #     self.is_my_first_move = False
        #     if self.first_move is not None:
        #         if self.first_move in moves:
        #             return self.first_move

        # If there is no best obvious move, take consider of all possible moves
        best_score = float("-inf")
        best_move = None

        # Right now the depthest level it can reach is only max without min due to time
        depth = 0

        # possible move to add 1 grid, 2 grid and so on
        # possible_fill = [0, 0, 0, 0, 0]
        # for move in moves:
        #     tile_grab = move[2]
        #     if tile_grab.num_to_floor_line == 0:
        #         possible_fill[tile_grab.num_to_pattern_line - 1] += 1

        # actions_size = len(moves)
        # my_advance_actions = self.advance_get_available_actions(moves, game_state, self.id)
        # type_1_size = len(my_advance_actions[0])
        #
        # if type_1_size > 0.1 * actions_size:
        #     my_advance_actions = my_advance_actions[0]
        # else:
        #     my_advance_actions = my_advance_actions[0] + my_advance_actions[1] + my_advance_actions[2]

        # print(len(my_advance_actions))
        top_5_actions = self.greedy_agent.SelectTopMove(moves, game_state, 5)
        # top_5_actions = self.SelectTopMove(moves, game_state, 5)
        # print(top_5_actions)
        # print("------- top 5 start -----")
        # for action in top_5_actions:
        #     print((action[0], action[1], (
        #                 action[2].tile_type,
        #                 action[2].number,
        #                 action[2].pattern_line_dest,
        #                 action[2].num_to_pattern_line,
        #                 action[2].num_to_floor_line,
        #                 )
        #            )
        #           )
        # print("------- top 5 end -----")
        # print(top_5_actions)
        print(len(top_5_actions), top_5_actions)
        current_state = deepcopy(game_state)
        expected_score, _ = current_state.players[self.id].ScoreRound()

        for move in top_5_actions:
            print("top n:", move[0], move[1], seeTile(move[2]), self.greedy_agent.getQValue(game_state, move),
                  "v.s.", self.ExpectScoreAfterMyMove(move, game_state, expected_score))
            move_score = self.MiniMax(move, game_state, depth, self.id)
            print("   ", "action's minimax score:", move_score)
            if move_score > best_score:
                best_score = move_score
                best_move = move
        print("best: ", best_move[0], best_move[1], seeTile(best_move[2]), self.greedy_agent.getQValue(game_state, best_move),
              "v.s.", self.ExpectScoreAfterMyMove(best_move, game_state, expected_score))
        print("---------")
        return best_move

    def MiniMax(self, move: (Move, int, TileGrab), game_state: GameState, depth: int, player_id: int):
        current_state = deepcopy(game_state)
        expected_score, _ = current_state.players[self.id].ScoreRound()
        # finish diving when it is the highest depth or the end of the game
        if depth == 0:
            # return self.greedy_agent.getQValue(game_state, move)
            return self.ExpectScoreAfterMyMove(move, game_state, expected_score)

        # get the state after the move
        next_state = deepcopy(game_state)
        next_state.ExecuteMove(player_id, move)
        next_player_id = abs(player_id - 1)
        new_depth = depth - 1

        # check all possible moves for next player, if none, return current state value
        next_player_moves = next_state.players[next_player_id].GetAvailableMoves(next_state)
        if len(next_player_moves) == 0:
            # return self.greedy_agent.getQValue(game_state, move)
            if player_id == self.opponent_id:
                return False
            else:
                return self.ExpectScoreAfterMyMove(move, game_state, expected_score)

        # maximum score part
        if next_player_id == self.id:
            # actions_size = len(next_player_moves)
            # my_advance_actions = self.advance_get_available_actions(next_player_moves, next_state, self.id)
            # type_1_size = len(my_advance_actions[0])
            #
            # if type_1_size / actions_size > 0.1:

            # actions_size = len(next_player_moves)
            # my_advance_actions = self.advance_get_available_actions(next_player_moves, next_state, self.id)
            # type_1_size = len(my_advance_actions[0])
            #
            # if type_1_size > 0.1 * actions_size:
            #     my_advance_actions = my_advance_actions[0]
            # else:
            #     my_advance_actions = my_advance_actions[0] + my_advance_actions[1] + my_advance_actions[2]

            # print(len(my_advance_actions))
            top_5_actions = self.greedy_agent.SelectTopMove(next_player_moves, next_state, 5)
            # top_5_actions = self.SelectTopMove(next_player_moves, next_state, 5)

            best_score = float("-inf")
            for my_move in top_5_actions:
                move_score = self.MiniMax(my_move, next_state, new_depth, next_player_id)
                best_score = max(best_score, move_score)
            return best_score

        # minimal score part
        else:
            # assume opponent make a greedy choice
            # print("self.opponent_agent.SelectMove")
            opponent_move = self.opponent_agent.SelectMove(next_player_moves, next_state)
            move_score = self.MiniMax(opponent_move, next_state, new_depth, next_player_id)

            if move_score == False:
                return self.ExpectScoreAfterMyMove(move, game_state, expected_score)
            return move_score

    def SelectTopMove(self, moves: [(Move, int, TileGrab)], game_state: GameState, n=5):
        if len(moves) <= n:
            return moves

        hq = PriorityQueue()
        # print("###########")
        for m in moves:
            # print("###########", m)
            current_state = deepcopy(game_state)
            expected_score, _ = current_state.players[self.id].ScoreRound()
            q_value = self.ExpectScoreAfterMyMove(m, game_state, expected_score)
            # print("###########1", m)

            hq.push(m, q_value)

            if hq.count > n:
                hq.pop()
            # print("###########2", m)

        result = []
        # print("------- start ------")
        while not hq.isEmpty():
            result.append(hq.pop())
        # print("------- end ------")
        return result

    def ExpectScoreAfterMyMove(self, move: (Move, int, TileGrab), game_state: GameState, neighbour_bonus: float):
        """
        The most significant function that calculate the score for the move
        """
        state_after_move = deepcopy(game_state)
        # print((move[0], move[1], seeTile(move[2])))
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
        # TODO changes of bonus
        bonus = my_state.EndOfGameScore()
        if self.turn <= self.TURN_IGNORE_FINAL_BONUS:
            final_score += bonus * self.TURN_IGNORE_FINAL_BONUS
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

        # # Feature 7: If there are some same tiles with same amount in the factory or in the center
        # # we don't need to fill the current tile so hurry
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
        #
        # # Feature 8: give bonus to the pattern line that already have some tile
        # if tile_grab.pattern_line_dest != -1:
        #     final_score += line_n[tile_grab.pattern_line_dest] * 0.0001


        return final_score


def seeTile(tile_grab: TileGrab):
    print(
        [tile_grab.tile_type,
         tile_grab.number,
         tile_grab.pattern_line_dest,
         tile_grab.num_to_pattern_line,
         tile_grab.num_to_floor_line])


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (priority, _, item) = heapq.heappop(self.heap)
        # print(priority, item)
        return item

    def pop_priority_item(self):
        (priority, _, item) = heapq.heappop(self.heap)
        # print(priority, item)
        return priority, item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class GreedyAgent(AdvancePlayer):
    IGNORE_BONUS_THRESHOLD = 3

    # initialize
    # The following function should not be changed at all
    def __init__(self, _id):
        super().__init__(_id)
        self.weights = [1, -0.4, 0.015, 0.01]
        self.curr_round = -1

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state: GameState):
        self.curr_round += 1
        print("---------- round", self.curr_round, "start ------------")
        return None

    def SelectTopMove(self, moves: [(Move, int, TileGrab)], game_state: GameState, n=5):
        if len(moves) <= n:
            return moves

        hq = PriorityQueue()
        # print("###########")
        for m in moves:
            # print("###########", m)
            q_value = self.getQValue(game_state, m)
            # print("###########1", m)

            hq.push(m, q_value)

            if hq.count > n:
                hq.pop()
            # print("###########2", m)

        result = []
        # print("------- start ------")
        while not hq.isEmpty():
            result.append(hq.pop())
        # print("------- end ------")
        return result

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated,
    # a random action will be selected, and you will receive
    # a timeout warning
    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        # move[1] is factory ID that illustrate the source of tile, -1 for center
        # find the action with highest Q value
        maxQ = float("-inf")
        curr_max = None

        for m in moves:
            q_value = self.getQValue(game_state, m)
            if q_value > maxQ:
                maxQ = q_value
                curr_max = m

        ns = self.getNextState(game_state, curr_max)

        # print("   ", self.id, ":", self.featureExtractor(game_state, curr_max))
        # print("   ", "this:", self.expectScore(game_state), " that:", self.expectScore(ns))
        # print("")
        return curr_max

    def getQValue(self, game_state: GameState, action) -> float:
        """get the Q value for a specify state with the performed action"""
        q_value = 0.0
        features = self.featureExtractor(game_state, action)

        for i in range(len(self.weights)):
            q_value += self.weights[i] * features[i]

        return q_value

    def featureExtractor(self, game_state: GameState, move: (Move, int, TileGrab)) -> list:
        """
        return the feature that we extract from a specific game state
        that can be used for Q value calculation

        feature = [expected score, bonus at end]

        :return a dictionary that contains the value we want to use in this game
        """
        features = []
        # TODO: As now next state won't be used for any other purpose, to save time, no deepcopy
        next_state = self.getNextState(game_state, move)
        expect_gain = self.expectGain(game_state, next_state)

        # expected score for the current action exec
        if self.curr_round < self.IGNORE_BONUS_THRESHOLD:

            # TODO: examine this field
            # suppose 90% of game end in 5 rounds
            if move[0] == Move.TAKE_FROM_CENTRE and not game_state.first_player_taken and \
                    self.curr_round < 4:
                # get first player token
                expect_gain += 1

            # only ignore the positive mark
            if expect_gain > 0:
                expect_gain *= 0.1

            features.append(expect_gain)
        else:
            features.append(expect_gain)

        # penalise add only a few grad to a long pattern
        tile_grab: TileGrab = move[-1]
        line_n = game_state.players[self.id].lines_number

        if tile_grab.pattern_line_dest != -1:
            if line_n[tile_grab.pattern_line_dest] == 0:
                # total capacity - tile already have - # we going to add
                remains = (tile_grab.pattern_line_dest + 1) - tile_grab.num_to_pattern_line
                features.append(remains)
            else:
                features.append(0)
        else:
            features.append(0)

        # give a slightly higher point to collect more
        features.append(move[-1].num_to_pattern_line)

        # give bonus to the pattern line that already have some tile
        if tile_grab.pattern_line_dest != -1:
            features.append(line_n[tile_grab.pattern_line_dest])
        else:
            features.append(0)

        return features

    def getNextState(self, game_state: GameState, action) -> GameState:
        """give a state and action, return the next state"""
        next_state: GameState = deepcopy(game_state)
        next_state.ExecuteMove(self.id, action)
        return next_state

    def expectGain(self, curr_state, next_state):
        copy_curr = deepcopy(curr_state)
        curr_expected_score, curr_bonus = self.expectScore(copy_curr)
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

