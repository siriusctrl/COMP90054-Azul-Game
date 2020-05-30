"""
Author:      XuLin Yang
Student id:  904904
Description: monte carlo tree search agent
             modify from https://www.moderndescartes.com/essays/deep_dive_mcts/
"""
import itertools
import sys
from copy import deepcopy

sys.path.append("players/StaffTeamEasy")
import os

from advance_model import *
import json
from math import sqrt, log

from players.StaffTeamEasy.environment import OpponentNode, OpponentPolicy, simulation, score_reward, random_simulation_policy, delta_score_reward
from players.StaffTeamEasy.util import get_opponent_player_id, getNextState

N_ITER = "n_iter"
CP = "Cp"
N_DEPTH = "n_depth"
GAMMA = "gamma"


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
        # self.action_next_state = {}  # {move: opponent's game_state}
        # self.opponent_actions = {}   # {move: [opponent move]}
        # self.children = {}           # {(move, opponent move): UCTNode}
        self.action_children = {}    # {move: [UCTNode]}
        # update all nodes in children and opponent_nodes
        # for action in self.actions:
        #     opponent_node = OpponentNode(self.getNextState(self.game_state, action, self.player_id), self.policy, self.player_id, self.opponent_id)
        #     self.opponent_nodes[action] = opponent_node
        #     self.children[action] = [UCTNode(s, self.player_id, self.opponent_id, self.policy, action, self) for s in opponent_node.get_next_states()]

        # self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int

        # used mcts with bellman equation back propagation
        self.vs = 0

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

    def q_value(self, action):
        """
        :return: float, q_value = avg_reward = total_reward / visit_count
        """
        print("q_value")
        result = self.total_value / (1 + self.number_visits)  # add 1 to avoid zero division
        print(result)
        # TODO check my calculation
        return result

    def u_value(self, agent):
        print("u_value")
        # TODO check my calculation
        result = 2 * agent.cp * sqrt(2 * log(self.parent.number_visits) / (1 + self.number_visits))  # add 1 to avoid zero division
        print(result)
        return result
        # if self.parent:
        #     return 2 * agent.cp * sqrt(2 * log(self.parent.number_visits) / (1 + self.number_visits))  # add 1 to avoid zero division
        # else:
        #     return 0

    def best_child(self, agent):
        """
        :param agent:
        :return: best child UCTNode based on 'max(q_value) + u_value'
        """
        cur_min = -float("inf")
        result = None
        # for child in self.children.values():
        #     child_value = child.total_value / (1 + child.number_visits) + 2 * agent.cp * sqrt(2 * log(self.number_visits+1) / (1 + child.number_visits))
        #     if child_value > cur_min:
        #         result, cur_min = child, child_value
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
            # print("error: null child")
            return self, False
        else:
            # print("success")
            pass
        return result, True

    def select_leaf(self, agent):
        """
        :param agent: UCTAgent
        :return:
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
            # print("no expansion")
            pass

    def add_child(self, action):
        if action in self.action_children:
            # print("    |", action)
            for child in self.action_children[action]:
                child.initialize()
        else:
            print("error", action)

    # ************************** step 3: simulation *************************************************************************************
    def simulate(self):
        """
        :return: float: reward
        """
        # TODO have a simulation function in environment.py,
        #  1. simulate from current node until scoring phase and have rank as reward (a progress in rank gives +1, keep rank and higher score gives +1,
        #  keep rank and lower score gives +0.5, etc)
        #  2. or simulate for N moves and if reach scoring phase before N moves then use the reward above otherwise use a reward for pattern line (....)
        #  3. how to deal with opponent's action? assume opponent choose the greedy action?
        # return simulation(self.game_state, self.player_id, self.opponent_id, random_simulation_policy, score_reward)

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

        BRANCHING_FACTOR_THRESHOLD = 10

        if branching_factor > BRANCHING_FACTOR_THRESHOLD:
            return self.greedyAgent.SelectMove(moves, game_state)
            # return random.choice(moves)

        root = UCTNode(game_state=game_state, player_id=self.id, opponent_id=self.opponent_id, policy=self.policy,
                       parent_opponent_action=None, parent_move=None, parent=None, moves=moves)
        root.initialize()

        counter = 0
        N_ITER = 200

        elapsed = (time.time() - start)
        timeout_threshold = timeout - 0.1 - elapsed
        # print("timeout_threshold:", timeout_threshold, "branching factor: ", branching_factor)

        while True:
            # print("    ", "myPlayer iter =", counter)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                # print("timeout")
                break

            leaf = root.select_leaf(self)
            if not leaf:
                print("#####", "error select leaf", "#####", moves)
                return random.choice(moves)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                # print("timeout")
                break
            # print("    ", "    select finished, used:", elapsed)

            leaf.expand()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                # print("timeout")
                break
            # print("    expand finished, used:", elapsed)

            simulated_reward = leaf.simulate()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                # print("timeout")
                break
            # print("    ","    simulation finished", simulated_reward, ", used:", elapsed)

            leaf.back_propagate(simulated_reward)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                # print("timeout")
                break
            # print("    ", "    back propagate finished, used:", elapsed)

            counter += 1
        print("##########", counter, "##########")
        cur_min = -float("inf")
        next_action = None
        for action, child in root.action_children.items():
            child_value = 0  # sum(child, lambda x: x.total_value)
            for c in child:
                if c.number_visits > 0:
                    child_value += c.total_value / c.number_visits
                # print("    "*2, c.total_value, c.number_visits)

            if child_value > cur_min:
                cur_min, next_action = child_value, action

        print(next_action, cur_min)
        elapsed = (time.time() - start)
        print("##########", counter, "##########", "total used: ", elapsed, "branching_factor: ", branching_factor)
        return next_action


class GreedyAgent(AdvancePlayer):
    # initialize
    # The following function should not be changed at all
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.weights = [1, -0.4]
        # decay for Q value
        self.discount = 0.9
        # learning rate for the Q value
        self.alpha = 0.2
        # exploration and exploitation rule for epsilon greedy
        self.epsilon = 1
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

        # print(maxQ)
        # if self.flipCoin():
        #     return curr_max
        # else:
        #     return random.choice(moves)

        return curr_max

    def getQValue(self, game_state: GameState, action) -> float:
        """get the Q value for a specify state with the performed action"""
        q_value = 0.0
        features = self.featureExtractor(game_state, action)

        for i in range(len(self.weights)):
            q_value += self.weights[i] * features[i]

        return q_value

    def update(self, game_state: GameState, action) -> None:
        """
            update weight, this will be called at the beginning of each state to
            update the parameters for previous state and action
        """
        next_state = self.getNextState(game_state, action)
        reward = 0

    def featureExtractor(self, game_state: GameState, move: (Move, int, TileGrab)) -> list:
        """
        return the feature that we extract from a specific game state
        that can be used for Q value calculation

        feature = [expected score, bonus at end]

        :return a dictionary that contains the value we want to use in this game
        """
        features = []
        next_state = self.getNextState(game_state, move)

        # expected score for the current action exec
        features.append(self.expectGain(game_state, next_state))

        # penalise add only a few grad to a long pattern
        tile_grab: TileGrab = move[-1]
        line_n = game_state.players[self.id].lines_number
        # total capacity - tile already have - # we going to add
        remains = (tile_grab.pattern_line_dest + 1) - line_n[tile_grab.pattern_line_dest] \
                  - tile_grab.num_to_pattern_line
        features.append(remains)

        return features

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
        return True if random.random() < self.epsilon else False
