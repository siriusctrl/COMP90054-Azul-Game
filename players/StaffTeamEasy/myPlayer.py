"""
Author:      XuLin Yang
Student id:  904904
Description: monte carlo tree search agent
"""
import heapq
import sys
from copy import deepcopy

sys.path.append("players/StaffTeamEasy")

from advance_model import *
from math import sqrt, log

from players.StaffTeamEasy.environment import OpponentPolicy, simulation, random_simulation_policy, delta_score_reward
from players.StaffTeamEasy.util import get_opponent_player_id, getNextState

class UCTNode:
    def __init__(self, game_state: GameState, player_id, opponent_id, policy: OpponentPolicy,
                 parent_opponent_action: [(Move, int, TileGrab)], parent_move=None, parent=None, prior=0, moves=None):
        self.player_id = player_id
        self.opponent_id = opponent_id
        self.policy = policy

        self.game_state = game_state
        self.is_expanded = False
        self.parent: UCTNode = parent
        self.parent_move = parent_move
        self.parent_opponent_action = parent_opponent_action

        self.actions = moves
        self.action_children = {}    # {move: [UCTNode]}

        # self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int

    def initialize(self):
        for action in self.get_actions():
            opponent_state = getNextState(self.game_state, action, self.player_id)
            # self.action_next_state[action] = opponent_state

            opponent_actions = opponent_state.players[self.opponent_id].GetAvailableMoves(opponent_state)
            # self.opponent_actions[action] = opponent_actions

            action_children = []
            for opponent_action in opponent_actions:
                child_game_state = getNextState(opponent_state, opponent_action, self.opponent_id)
                uct_node = UCTNode(game_state=child_game_state, player_id=self.player_id, opponent_id=self.opponent_id, policy=self.policy,
                                   parent_opponent_action=opponent_action, parent_move=action, parent=self)

                # self.children[(action, opponent_action)] = uct_node
                action_children.append(uct_node)

            self.action_children[action] = action_children

    def get_actions(self):
        """
        lazy evaluation
        """
        if not self.actions:
            self.actions = self.game_state.players[self.player_id].GetAvailableMoves(self.game_state)
        return self.actions

    # ************************** step 1: selection *************************************************************************************
    def best_child(self, agent):
        """
        :param agent:
        :return: best child UCTNode based on 'max(q_value) + u_value'
        """
        cur_min = -float("inf")
        result = None
        for child in self.action_children.values():
            for c in child:
                if c.number_visits == 0:
                    result, cur_min = c, float("inf")
                    return result, True
                child_value = c.total_value / (1 + c.number_visits) + 2 * agent.cp * sqrt(2 * log(self.number_visits+1) / (1 + c.number_visits))
                if child_value > cur_min:
                    result, cur_min = c, child_value

        # no child, so return self with False
        if not result:
            return self, False
        else:
            pass
        return result, True

    def select_leaf(self, agent):
        """
        depth limited expansion
        """
        current: UCTNode = self
        cur_depth = 0
        has_child = True

        while current and has_child and current.is_expanded and cur_depth < agent.n_depth:
            current, has_child = current.best_child(agent)
            # if not has_child:
            #     print(cur_depth)

            cur_depth += 1
        return current

    # ************************** step 2: expansion *************************************************************************************

    def expand(self):
        if not self.is_expanded:
            self.is_expanded = True
            for action in self.get_actions():
                self.add_child(action)
        else:
            pass

    def add_child(self, action):
        if action in self.action_children:
            for child in self.action_children[action]:
                child.initialize()
        else:
            print("error", action)

    # ************************** step 3: simulation *************************************************************************************
    def simulate(self):
        """
        :return: float: reward
        """

        return simulation(self.game_state, self.player_id, self.opponent_id, random_simulation_policy, delta_score_reward)

    # ************************** step 4: back-propagation *************************************************************************************
    def back_propagate(self, simulated_reward):
        current = self
        current.number_visits += 1
        current.total_value += simulated_reward

        while current.parent:
            current = current.parent
            current.number_visits += 1
            current.total_value += simulated_reward


class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)

        # read from setup file
        self.n_iter = 100
        self.cp = 0.7071
        self.n_depth = 3
        self.gamma = 0.9

        self.opponent_id = None
        self.policy = OpponentPolicy()

        self.greedyAgent = GreedyAgent(self.id)
        # print("myPlayer __init__ finish")

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state: GameState):
        self.opponent_id = get_opponent_player_id(game_state, self.id)
        # print("myPlayer StartRound finish")
        return None

    # Each player is given 1 second to select next best move
    # If exceeds 5 seconds, all your code will be terminated,
    # a random action will be selected, and you will receive
    # a timeout warning
    def SelectMove(self, moves: [(Move, int, TileGrab)], game_state: GameState):
        """
        :param moves: list of (Move, from which factory; -1 for center, Grab effect)
        :param game_state: current game state
        :return:
        """
        start = time.time()
        timeout = 1
        branching_factor = len(moves)

        BRANCHING_FACTOR_THRESHOLD = 6

        if branching_factor > BRANCHING_FACTOR_THRESHOLD:
            return self.greedyAgent.SelectMove(moves, game_state)

        # print to draw diagrams
        # print("branching factor: ", branching_factor)

        root = UCTNode(game_state=game_state, player_id=self.id, opponent_id=self.opponent_id, policy=self.policy,
                       parent_opponent_action=None, parent_move=None, parent=None, moves=moves)
        root.initialize()

        counter = 0

        elapsed = (time.time() - start)
        timeout_threshold = timeout - 0.2 - elapsed

        while True:
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                break

            leaf = root.select_leaf(self)
            if not leaf:
                print("#####", "error select leaf", "#####", moves)
                return random.choice(moves)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                break

            leaf.expand()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                break

            simulated_reward = leaf.simulate()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                break
            leaf.back_propagate(simulated_reward)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                break

            counter += 1

        cur_min = -float("inf")
        next_action = None
        for action, child in root.action_children.items():
            child_value = 0
            for c in child:
                if c.number_visits > 0:
                    child_value += c.total_value / c.number_visits

            if child_value > cur_min:
                cur_min, next_action = child_value, action

        elapsed = (time.time() - start)

        # print to draw diagrams
        # print("time elapsed:", elapsed)
        # print("n_ter:", counter)
        return next_action


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
        print("---------- round", self.curr_round, "start ------------")
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
