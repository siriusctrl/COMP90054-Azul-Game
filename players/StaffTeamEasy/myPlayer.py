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
    ROUND_IGNORE_NEIGHBOUR_BONUS = 3
    ROUND_IGNORE_FINAL_BONUS = 3

    def __init__(self, _id):
        super().__init__(_id)
        self.curr_round = 0

        # weight settings, minimal scale of weight is 0.00001
        self.neighbour_weight = 0.01
        self.final_bonus_weight = 0.4
        self.long_empty_penalty = -0.35
        self.long_fill_bonus = 0.01
        self.fill_bonus = 0.05
        self.turn = 0

        # the parameter for the first move of a turn
        self.is_my_first_move = False
        self.first_move = None

        self.opponent_id = abs(1 - self.id)
        self.opponent_agent = GreedyAgent(self.opponent_id)
        self.greedy_agent = GreedyAgent(self.id)

        self.top_move = None

    def StartRound(self, game_state):
        # we have an increasing focus on the bonus as the game goes
        self.curr_round += 1
        self.turn = 0

        return None

    def SelectMove(self, moves, game_state):
        # start = time.clock()
        try:
            self.turn += 1
            self.greedy_agent.curr_round = self.curr_round
            # select move for the first two turn as there is not much extra info minmax can provide
            # but by doing this can prevent timeout
            if self.turn < 2:
                return self.greedy_agent.SelectMove(moves, game_state)
            self.top_move = None

            # this function normally needs extra 0.1-0.2s invocation time, reduce computational time to fit
            # the gap
            return func_timeout(0.7, self.SelectMiniMaxMove, args=(moves, game_state))
        except FunctionTimedOut:
            # print("Time-out", time.clock() - start)
            if self.top_move is not None:
                return self.top_move
            else:
                return random.choice(moves)

    def SelectMiniMaxMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        depth = 2
        top_5_actions = self.greedy_agent.SelectTopMove(moves, game_state, 5)

        self.top_move = top_5_actions[-1]

        current_state = deepcopy(game_state)
        expected_score, _ = current_state.players[self.id].ScoreRound()
        expected_bonus = current_state.players[self.id].EndOfGameScore()
        opponent_score, _ = current_state.players[self.opponent_id].ScoreRound()
        opponent_bonus = current_state.players[self.opponent_id].EndOfGameScore()
        chase = (expected_score + expected_bonus) < (opponent_score + opponent_bonus)

        candidates = []

        for move in top_5_actions:
            diff, gain = self.MiniMax(move, game_state, depth, self.id, expected_score + expected_bonus)
            candidates.append((diff, gain, move))

        if chase:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[-1][-1]
        else:
            candidates.sort(key=lambda x: (x[1], x[0]))
            return candidates[-1][-1]

    def MiniMax(self, move: (Move, int, TileGrab), game_state: GameState, depth: int, player_id: int, root_data: float):
        next_state = deepcopy(game_state)
        next_state.ExecuteMove(player_id, move)
        next_player_id = abs(player_id - 1)
        # finish diving when it is the highest depth or the end of the game

        if depth == 0:
            return self.expect_gain_with_root(next_state, root_data)

        new_depth = depth - 1

        # check all possible moves for next player, if none, return current state value
        next_player_moves = next_state.players[next_player_id].GetAvailableMoves(next_state)
        if len(next_player_moves) == 0:
            if player_id == self.opponent_id:
                return False
            else:
                return self.expect_gain_with_root(next_state, root_data)

        # maximum score part
        if next_player_id == self.id:
            top_5_actions = self.greedy_agent.SelectTopMove(next_player_moves, next_state, 5)

            best_score = (float("-inf"), float("-inf"))
            for my_move in top_5_actions:
                move_score = self.MiniMax(my_move, next_state, new_depth, next_player_id, root_data)

                if move_score > best_score:
                    best_score = move_score

            return best_score

        # minimal score part
        else:
            # assume opponent make a greedy choice
            opponent_move = self.opponent_agent.SelectMove(next_player_moves, next_state)
            move_score = self.MiniMax(opponent_move, next_state, new_depth, next_player_id, root_data)

            if not move_score:
                return self.expect_gain_with_root(next_state, root_data)
            return move_score

    def expect_gain_with_root(self, curr_state: GameState, root_data: float) -> (float, float):
        copy_curr: GameState = deepcopy(curr_state)
        curr_player = copy_curr.players[self.id]
        expect_score, _ = curr_player.ScoreRound()
        expect_bonus = curr_player.EndOfGameScore()

        curr_opponent = copy_curr.players[self.opponent_id]
        op_score, _ = curr_opponent.ScoreRound()
        op_bonus = curr_opponent.EndOfGameScore()

        return expect_score + expect_bonus - op_score - op_bonus, expect_score + expect_bonus - root_data

    def SelectTopMove(self, moves: [(Move, int, TileGrab)], game_state: GameState, n=10):
        if len(moves) <= n:
            return moves

        hq = PriorityQueue()
        for m in moves:
            current_state = deepcopy(game_state)
            expected_score, _ = current_state.players[self.id].ScoreRound()
            q_value = self.greedy_agent.getQValue(game_state, m)

            hq.push(m, q_value)
            if hq.count > n:
                hq.pop()

        result = []

        while not hq.isEmpty():
            result.append(hq.pop())

        return result


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
        return item

    def pop_priority_item(self):
        (priority, _, item) = heapq.heappop(self.heap)
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
        return None

    def SelectTopMove(self, moves: [(Move, int, TileGrab)], game_state: GameState, n=5):
        if len(moves) <= n:
            return moves

        hq = PriorityQueue()
        for m in moves:
            q_value = self.getQValue(game_state, m)
            hq.push(m, q_value)

            if hq.count > n:
                hq.pop()

        result = []
        while not hq.isEmpty():
            result.append(hq.pop())
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
