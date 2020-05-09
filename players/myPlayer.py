from model import Player
import json
from math import sqrt, log


N_ITER = "n_iter"
CP = "Cp"
N_DEPTH = "n_depth"


class UCTNode:
    def __init__(self, game_state, move=None, parent=None, prior=0):
        self.game_state = game_state
        self.is_expanded = False
        self.parent = parent
        self.move = move
        self.children = {}  # Dict[move, UCTNode]
        # self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int

    # ************************** step 1: selection *************************************************************************************

    def q_value(self):
        """
        :return: float, q_value = avg_reward = total_reward / visit_count
        """
        # TODO check my calculation
        return self.total_value / (1 + self.number_visits)  # add 1 to avoid zero division

    def u_value(self, agent):
        # TODO check my calculation
        return 2 * agent.cp * sqrt(2 * log(self.parent.number_visits) / (1 + self.number_visits))  # add 1 to avoid zero division

    def best_child(self, agent):
        """
        :param agent:
        :return: best child UCTNode based on 'max(q_value) + u_value'
        """
        return max(self.children.values(), key=lambda node: node.q_value() + node.u_value(agent))

    def select_leaf(self, agent):
        current = self
        while current.is_expanded:
            current = current.best_child(agent)
        return current

    # ************************** step 2: expansion *************************************************************************************

    def expand(self):
        self.is_expanded = True
        # TODO finish this; not sure the provided GameState can apply action and give the next GameState
        for action in self.game_state:
            self.add_child(action, next_state)

    def add_child(self, action, next_state):
        self.children[action] = UCTNode(next_state, action, self)

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
        return 0

    # ************************** step 4: back-propagation *************************************************************************************
    def back_propagate(self, simulated_reward):
        current = self
        while current.parent:
            current.number_visits += 1
            current.total_value += simulated_reward
            current = current.parent


class UCTAgent(Player):
    def __init__(self, _id):
        super().__init__(_id)

        # read from setup file
        self.n_iter = 0
        self.cp = None
        self.n_depth = None
        with open('mcts.json') as file:
            data = json.load(file)
            self.n_iter = data[N_ITER]
            self.cp = data[CP]
            self.n_depth = data[N_DEPTH]

    def SelectMove(self, moves, game_state):
        root = UCTNode(game_state)
        for _ in range(self.n_iter):
            leaf = root.select_leaf(self)
            leaf.expand()
            simulated_reward = leaf.simulate()
            leaf.back_propagate(simulated_reward)

        return max(root.children.items(), key=lambda item: item[1].number_visits)
