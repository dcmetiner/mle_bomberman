import os
import pickle
import random
from .q_utilities import *

import numpy as np


ACTIONS = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'BOMB', 'WAIT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.STATE_SIZE = get_state_space_size()
    self.MODEL_NAME = f"q_table_{self.STATE_SIZE}.npy"

    self.EPSILON = 0.1
    self.EPSILON_MIN = 0.005
    self.ALPHA = 0.1
    self.GAMMA = 0.9

    if not os.path.isfile( self.MODEL_NAME):
        self.logger.info("Setting up model from scratch.")
        self.q_table = np.array([[0 for col in range(6)] for row in range(self.STATE_SIZE)], dtype=np.float64)
        np.save( self.MODEL_NAME, self.q_table)

    else:
        self.q_table = np.load(self.MODEL_NAME)


def rotate_back_q_table(q_table, rot):
    for i in range(rot):
        q_table = np.concatenate(([q_table[3]], q_table[:3], q_table[4:]))
    return q_table


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.feature_vector = state_to_feature_vector(game_state)
    self.state_id, rotation_cnt = feature_vector_to_id(self.feature_vector)
    self.rot_cnt = rotation_cnt



    # todo Exploration vs exploitation

    decision_factor = None

    if  self.train and random.uniform(0, 1) < self.EPSILON:
        decision_factor = "RANDOMLY"
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_id = np.random.choice(range(6), p=[.2, .2, .2, .2, .1, .1])
    else:
        decision_factor = "from Q-TABLE"

        # print_state(game_state)
        # state_id = feature_vector_to_id(feature_vector)
        action_id = np.argmax(self.q_table[self.state_id, :])
    self.action_before_rotation = ACTIONS[action_id]

    action_final = rotate_back_n_times(self.action_before_rotation, rotation_cnt)

    print_state(game_state, logger=self.logger.debug)
    self.logger.debug(f"feature vector: {self.feature_vector}")
    rotated, rot_cnt = canonicalize_neighbours(self.feature_vector[:4])
    self.logger.debug(f"converted feature vector: {rotated+self.feature_vector[4:]}, with {str(rot_cnt)} rotations")
    self.logger.debug(f"Q table results {rotate_back_q_table(self.q_table[self.state_id, :], rotation_cnt)}")
    # self.logger.debug(f"Action taken {decision_factor}: {self.action_before_rotation} (act_id:{str(action_id)}), and final action after rotation: {action_final}")
    self.logger.debug(f"final action after rotation: {action_final}")

    return action_final



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
