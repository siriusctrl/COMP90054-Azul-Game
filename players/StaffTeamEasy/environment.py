"""
Author:      XuLin Yang
Student id:  904904
Date:        2020-5-22 01:32:19
Description: environment for azul
"""
import random

from model import GameState, Move, TileGrab, PlayerState
from players.StaffTeamEasy.util import Counter, sample, getNextState


class OpponentPolicy:
    @staticmethod
    def get_action_prob(game_state: GameState, actions: [(Move, int, TileGrab)]) -> Counter:
        """
        uniform as default
        :param game_state:
        :param actions: game_state's playing player actions
        :return:
        """
        counter = Counter()
        for action in actions:
            counter[action] = 1
        counter.normalize()
        return counter


class MDP:
    # def __init__(self, game_state: GameState, game_state_player_id: int, policy: Policy):
    #     self.game_state = game_state
    #     self.game_state_player_id = game_state_player_id
    #     self.game_state_player_state = game_state.players[self.game_state_player_id]
    #     self.game_state_actions = self.game_state_player_state.GetAvailableMoves(self.game_state)
    #     self.policy = policy
    #
    # def get_action_prob(self):
    #     """
    #     default is uniform prob
    #     """
    #     self.policy.get_action_prob(self.game_state, self.game_state_actions)

    @staticmethod
    def get_action_prob_static(game_state: GameState, game_state_player_actions: [(Move, int, TileGrab)], policy: OpponentPolicy) -> Counter:
        return policy.get_action_prob(game_state, game_state_player_actions)


class OpponentNode:
    def __init__(self, opponent_game_state: GameState, policy: OpponentPolicy, my_player_id: int, opponent_player_id: int):
        """
        the context of the attributes in this class is in self player's perspective
        :param opponent_game_state:
        :param my_player_id:
        :param opponent_player_id:
        :param policy:
        """
        self.opponent_game_state = opponent_game_state

        self.opponent_player_state: PlayerState = opponent_game_state.players[opponent_player_id]

        # opponent player's actions in its game state
        self.actions = self.opponent_player_state.GetAvailableMoves(self.opponent_game_state)
        # util.Counter({action: prob -> float})
        self.action_prob_counter = MDP.get_action_prob_static(self.opponent_game_state, self.actions, policy)
        # a cache version of {action: next_state}
        self.action_next_state = {action: None for action in self.actions}

        self.children_uct_node = {}

    def get_next_state_by_action(self, opponent_action: (Move, int, TileGrab)) -> GameState:
        """
        apply opponent_action to self.opponent_game_state
        :param opponent_action:
        :return:
        """
        if not self.action_next_state[opponent_action]:
            self.action_next_state[opponent_action] = self.opponent_player_state.GetAvailableMoves(self.opponent_game_state)
        return self.action_next_state[opponent_action]

    def sample_an_action(self):
        return sample(self.action_prob_counter)

    def get_prob_by_action(self, opponent_action: (Move, int, TileGrab)) -> float:
        return self.action_prob_counter[opponent_action]

    def get_next_states(self) -> [GameState]:
        result = []
        for action, next_state in self.action_next_state:
            if not next_state:
                self.action_next_state[action] = self.opponent_player_state.GetAvailableMoves(self.opponent_game_state)
                result.append(self.action_next_state[action])
            else:
                result.append(next_state)
        return result

    def set_children_uct_node_by_opponent_action(self, opponent_action: (Move, int, TileGrab), uct_node):
        self.children_uct_node[opponent_action] = uct_node


def simulation(game_state: GameState, player_id: int, opponent_id: int, simulation_policy, reward_function) -> float:
    cur_player_id = player_id
    next_player_id = opponent_id
    actions = game_state.players[cur_player_id].GetAvailableMoves(game_state)
    next_state = game_state

    # simulate until a turn end
    while actions:
        chosen_action = simulation_policy(next_state, actions, cur_player_id, next_player_id)

        next_state = getNextState(next_state, chosen_action, cur_player_id)
        cur_player_id, next_player_id = next_player_id = cur_player_id
        actions = next_state.players[cur_player_id].GetAvailableMoves(next_state)

    return reward_function(next_state, player_id, opponent_id)


def random_simulation_policy(game_state: GameState, actions: [(Move, int, TileGrab)], cur_player_id: int, next_player_id: int):
    return random.choice(actions)


def score_reward(game_state: GameState, player_id: int, opponent_id: int) -> float:
    player_score = game_state.players[player_id].score
    opponent_score = game_state.players[opponent_id].score

    if player_score > opponent_score:
        return 1
    elif player_score == opponent_score:
        return 0.2
    else:
        return -1
