"""
Author:      XuLin Yang
Student id:  904904
Description: monte carlo tree search agent
             modify from https://www.moderndescartes.com/essays/deep_dive_mcts/
"""
import itertools
import sys
sys.path.append("players/StaffTeamEasy")
import os

from advance_model import *
import json
from math import sqrt, log

from players.StaffTeamEasy.environment import OpponentNode, OpponentPolicy, simulation, score_reward, random_simulation_policy
from players.StaffTeamEasy.util import get_opponent_player_id, getNextState

N_ITER = "n_iter"
CP = "Cp"
N_DEPTH = "n_depth"
GAMMA = "gamma"


class UCTNode:
    def __init__(self, game_state: GameState, player_id, opponent_id, policy: OpponentPolicy,
                 parent_opponent_action: [(Move, int, TileGrab)], parent_move=None, parent=None, prior=0):
        self.player_id = player_id
        self.opponent_id = opponent_id
        self.policy = policy

        self.game_state = game_state
        self.is_expanded = False
        self.parent: UCTNode = parent
        self.parent_move = parent_move
        self.parent_opponent_action = parent_opponent_action

        self.actions = None
        self.opponent_nodes = {}  # {move: OpponentNode}
        self.children = {}  # {move: UCTNode}
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

    def get_actions(self):
        """
        lazy evaluation
        """
        if not self.actions:
            self.actions = self.game_state.players[self.player_id].GetAvailableMoves(self.game_state)
        return self.actions

    # def get_children_by_action(self, action) -> [GameState]:
    #     return self.opponent_nodes[action].get_next_states()
    #
    # def get_all_children(self) -> [GameState]:
    #     for action in self.actions:
    #         if action not in self.opponent_nodes:
    #             self.opponent_nodes[action] = OpponentNode(self.getNextState(self.game_state))

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
        # print("@@@@@@@", "best_child")
        cur_min = -float("inf")
        result = None
        for children in self.children.values():
            for child in children:
                # print("!!!", child, child.total_value, child.number_visits, child.parent.number_visits)
                # print("!!!", child.total_value / (1 + child.number_visits))
                # print("!!!", 2 * agent.cp * sqrt(2 * log(self.number_visits) / (1 + child.number_visits)))
                child_value = child.total_value / (1 + child.number_visits) + 2 * agent.cp * sqrt(2 * log(self.number_visits) / (1 + child.number_visits))
                # print("!!!", child_value, cur_min, child_value > cur_min)
                if child_value > cur_min:
                    result, cur_min = child, child_value
        # result = max(list(itertools.chain(*self.children.values())), key=lambda node: node.q_value() + node.u_value(agent))
        # print(result)
        return result

    def select_leaf(self, agent):
        """
        :param agent: UCTAgent
        :return:
        """
        current = self

        while current.is_expanded:
            # print("########")
            current = current.best_child(agent)
            # print("########", current)
            if not current:
                return None
        return current

    def bellman_equation(self, action, agent):
        """
        tmp method here in case that we want to switch to bellman_equation update in back propagation
        :param action:
        :param agent:
        :return:
        """
        result = 0
        opponent_node: OpponentNode = self.opponent_nodes[action]
        for opponent_action in opponent_node.actions:
            # r(s, action, s': uct child) = 0 here
            result += opponent_node.get_prob_by_action(opponent_action) * (0 + agent.gamma * opponent_node.children_uct_node[opponent_action].vs)
        return result

    # ************************** step 2: expansion *************************************************************************************

    def expand(self):
        self.is_expanded = True
        for action in self.get_actions():
            self.add_child(action)

    def add_child(self, action):
        if action not in self.children:
            # print("    |", action)
            opponent_node = OpponentNode(opponent_game_state=getNextState(self.game_state, action, self.player_id),
                                         policy=self.policy,
                                         my_player_id=self.player_id,
                                         opponent_player_id=self.opponent_id)
            self.opponent_nodes[action] = opponent_node
            self.children[action] = []
            for opponent_action in opponent_node.actions:
                # print("        |", opponent_action)
                s = opponent_node.get_next_state_by_action(opponent_action)
                uct_node = UCTNode(game_state=s, player_id=self.player_id, opponent_id=self.opponent_id, policy=self.policy,
                                   parent_opponent_action=opponent_action, parent_move=action, parent=self)

                self.children[action].append(uct_node)
                opponent_node.set_children_uct_node_by_opponent_action(opponent_action, uct_node)

    # def add_opponent_node(self, action, next_state):
    #     self.opponent_nodes[action] = OpponentNode(next_state, self.policy, self.id, self.opponent_id)

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
        return simulation(self.game_state, self.player_id, self.opponent_id, random_simulation_policy, score_reward)

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
        self.n_depth = 1
        self.gamma = 0.9
        # with open('mcts.json') as file:
        #     data = json.load(file)
        #     self.n_iter = data[N_ITER]
        #     self.cp = data[CP]
        #     self.n_depth = data[N_DEPTH]
        #     self.gamma = data[GAMMA]
        self.opponent_id = None
        self.policy = OpponentPolicy()
        print("myPlayer __init__ finish")

    # Each player is given 5 seconds when a new round started
    # If exceeds 5 seconds, all your code will be terminated and
    # you will receive a timeout warning
    def StartRound(self, game_state: GameState):
        self.opponent_id = get_opponent_player_id(game_state, self.id)
        print("myPlayer StartRound finish")
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
        timeout = 10
        timeout_threshold = timeout - 0.2
        start = time.time()
        root = UCTNode(game_state=game_state, player_id=self.id, opponent_id=self.opponent_id, policy=self.policy,
                       parent_opponent_action=None, parent_move=None, parent=None)
        counter = 0
        N_ITER = 3
        while counter < N_ITER:
            print("myPlayer iter =", counter)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                print("timeout")
                break

            leaf = root.select_leaf(self)
            if not leaf:
                return random.choice(moves)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                print("timeout")
                break
            print("    select finished, used:", elapsed)

            leaf.expand()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                print("timeout")
                break
            print("    expand finished, used:", elapsed)

            simulated_reward = leaf.simulate()
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                print("timeout")
                break
            print("    simulation finished", simulated_reward, ", used:", elapsed)

            leaf.back_propagate(simulated_reward)
            elapsed = (time.time() - start)
            if elapsed >= timeout_threshold:  # prevent 1 second timeout
                print("timeout")
                break
            print("    back propagate finished, used:", elapsed)

            counter += 1
        print("##########")
        # print(list(root.children.items()))
        cur_min = -float("inf")
        next_action = None
        # print(len(root.children))
        for action, child in root.children.items():
            child_value = 0  # sum(child, lambda x: x.total_value)
            for c in child:
                child_value += c.total_value

            # print(cur_min, next_action, action, child_value, child)
            if child_value > cur_min:
                cur_min, next_action = child_value, action

        # next_action = max(list(itertools.chain(*root.children.items())), key=lambda item: item[1].total_value)
        print(next_action)
        print("##########")
        return next_action
