

from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, rotate_back_q_table
import numpy as np

from q_learning_train import state_to_id
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

STATE_SIZE = 360
MODEL_NAME = f"q_table_{STATE_SIZE}.npy"

ALPHA = 0.1
GAMMA = 0.95

ACTIONS = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'BOMB', 'WAIT']
ACTION_TO_ID = {action : id for id, action in enumerate(ACTIONS)}

LEFT_BOMB_FALSELY = "LEFT_BOMB_WITHOUT_POS_EFFECT"
ACTION_TAKEN_CAUSING_SELF_KILL = "ACTION_TAKEN_CAUSING_SELF_KILL"
WAIT_THOUGH_SAFE_BETTER_ACTIONS = "WAIT_THOUGH_SAFE_BETTER_ACTIONS"
GOOD_ACTION = "GOOD_ACTION"
BOMB_LEFT_CORRECTLY1 = "BOMB_LEFT_WISELY1"
BOMB_LEFT_CORRECTLY2 = "BOMB_LEFT_WISELY2"
BOMB_LEFT_CORRECTLY3 = "BOMB_LEFT_WISELY3"







def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.round_reward_sum = 0
    self.step_cnt = 0
    self.total_score = 0



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    action_id = ACTION_TO_ID[self_action]
    if (0 <= action_id < 4 and self.feature_vector[action_id] == 2) or (action_id==5 and self.feature_vector[4] == 4):
        events.append(ACTION_TAKEN_CAUSING_SELF_KILL)
        self.logger.warn("Action taken causing self kill")

    if action_id == 4 and self.feature_vector[4] == 0:
        events.append(LEFT_BOMB_FALSELY)
        self.logger.warn("left bomb without gain")

    if action_id == 5 and any([1==e for e in self.feature_vector[:4]]):
        events.append(WAIT_THOUGH_SAFE_BETTER_ACTIONS)
        self.logger.warn("wait though better move")

    if (0<=action_id<4 and self.feature_vector[action_id] == 1):
        events.append(GOOD_ACTION)
        self.logger.warn("GOOD ACTION")

    if  (action_id==4 and self.feature_vector[4] == 1):
        events.append(BOMB_LEFT_CORRECTLY1)
        self.logger.warn("Bomb left correctly")

    if  (action_id==4 and self.feature_vector[4] ==2):
        events.append(BOMB_LEFT_CORRECTLY2)
        self.logger.warn("Bomb left correctly")

    if  (action_id==4 and self.feature_vector[4] ==3):
        events.append(BOMB_LEFT_CORRECTLY3)
        self.logger.warn("Bomb left correctly")




    # old_state_id, _ = state_to_id(old_game_state)
    old_state_id = self.state_id
    assert old_state_id == self.state_id

    action_id = ACTION_TO_ID[self.action_before_rotation]
    new_state_id, _ = state_to_id(new_game_state)
    reward = reward_from_events(self, events)

    self.round_reward_sum += reward
    self.logger.debug(f"Old state id: {old_state_id}, action id: {action_id}, new state id: {new_state_id}")
    self.logger.debug(f"Reward: {reward}, total reward:  {self.round_reward_sum}")
    self.logger.debug(f"Updated q table: {rotate_back_q_table(self.q_table[old_state_id,:], self.rot_cnt)}")
    self.logger.debug("")
    self.logger.debug("***************************")
    self.logger.debug("")
    updated_q_val = (1 - ALPHA) * self.q_table[old_state_id, action_id] + ALPHA * (reward + GAMMA * self.q_table[new_state_id].max())
    self.q_table[old_state_id, action_id] = updated_q_val

    self.transitions.append(Transition(old_game_state, self_action, new_state_id, reward))





def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    last_state_id = self.state_id
    action_id = ACTION_TO_ID[self.action_before_rotation]
    reward = reward_from_events(self, events)
    self.round_reward_sum += reward
    self.logger.debug(f"Game END\nLast state id: {last_state_id}, action id: {action_id}")
    self.logger.debug(f"Reward: {reward}, total reward:  {self.round_reward_sum}")
    self.logger.debug(f"Updated q table: {rotate_back_q_table(self.q_table[last_state_id, :], self.rot_cnt)}")
    self.logger.debug("")
    self.logger.debug("***************************")
    self.logger.debug("")
    updated_q_val = (1 - ALPHA) * self.q_table[last_state_id, action_id] + ALPHA * (reward + GAMMA * 0) / 2
    self.q_table[last_state_id, action_id] = updated_q_val


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    np.save(self.MODEL_NAME, self.q_table)

    self.step_cnt += last_game_state['step']
    self.total_score += last_game_state['self'][1]

    if last_game_state['round'] % 100 == 0:
        print(f"100 Game details: Step count: {self.step_cnt}, total reward: {self.round_reward_sum}, round: {last_game_state['round']}, EPSILON: {self.EPSILON}, score: {self.total_score}")
        self.round_reward_sum = 0
        self.step_cnt = 0
        self.total_score = 0

    if last_game_state["round"] % 250 == 0:
        self.EPSILON *= 0.9
        self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 50,
        e.CRATE_DESTROYED: 3,
        e.INVALID_ACTION: -10,
        e.KILLED_SELF: -30,
        e.GOT_KILLED: -50,
        # e.WAITED: -4,
        # e.BOMB_DROPPED: 2,
        # e.MOVED_UP: 1,
        # e.MOVED_DOWN: 1,
        # e.MOVED_RIGHT: 1,
        # e.MOVED_LEFT: 1,
        # e.SURVIVED_ROUND: 1,
        # e.BOMB_EXPLODED: 0.1,
        # ACTION_TAKEN_CAUSING_SELF_KILL: -10,
        # LEFT_BOMB_FALSELY: -15,
        # WAIT_THOUGH_SAFE_BETTER_ACTIONS: -15,
        # GOOD_ACTION:2,
        # BOMB_LEFT_CORRECTLY1: 1,
        # BOMB_LEFT_CORRECTLY2: 3,
        # BOMB_LEFT_CORRECTLY3: 4



    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum