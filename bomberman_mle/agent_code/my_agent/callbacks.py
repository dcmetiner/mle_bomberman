import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
    self.epsilon =  0.01
    self.epsilon_decay = 0.995
    self.epsilon_values = []
    self.old_model = []
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = {}
    else:
        # if os.path.isfile("my-saved-model.pt"):
        #     with open("my-saved-model.pt", "rb") as file:
        #         self.old_model = pickle.load(file)
        
            # with open("my-saved-model-old.pt", "wb") as old_file:
            #     pickle.dump(old_model, old_file)

        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    # epsilon_decay = 0.995
    min_epsilon = 0.01
    # self.logger.debug(f"epsilon {self.epsilon}")
    if self.train and random.random() < self.epsilon:
        # self.logger.debug("Choosing action purely at random.")
        action_probabilities = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        action = np.random.choice(ACTIONS, p=action_probabilities)
    else:
        state_key = state_to_features(game_state)
        if state_key in self.model:
            q_values = self.model[state_key]
            action = ACTIONS[np.argmax(q_values)]
        else:
            action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    self.epsilon_values.append(self.epsilon)
    self.epsilon = max(min_epsilon, self.epsilon * self.epsilon_decay)

    # self.logger.debug("Querying model for action.")

    return action

# def state_to_features(game_state: dict) -> np.array:
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Converts the game state to the input of your model, i.e.
#     a feature vector.

#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.

#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """        
#     # print(game_state["round"])

#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None


#     # board = game_state['field']  # 2D array where -1=wall, 0=free space, 1=crate

#     # Example: Create a channel for bombs
#     bombs = np.zeros_like(board)
#     for bomb in game_state['bombs']:
#         bombs[bomb[0][0], bomb[0][1]] = bomb[1]  # Place bomb timer on the board

#     # Example: Create a channel for coins
#     coins = np.zeros_like(board)
#     for coin in game_state['coins']:
#         coins[coin[0], coin[1]] = 1  # Mark coin locations on the board

#     # Example: Encode opponents' positions
#     opponents = np.zeros_like(board)
#     for opponent in game_state['others']:
#         opponents[opponent[3][0], opponent[3][1]] = 1

#     # For example, you could construct several channels of equal shape, ...
#     channels = [board, bombs, coins, opponents]
#     # channels.append(...)
#     # concatenate them as a feature tensor (they must have the same shape), ...
#     stacked_channels = np.stack(channels)
#     # and return them as a vector
#     return stacked_channels.reshape(-1)

def state_to_features(game_state):
    """
    Convert the game state to a feature vector for the model.

    :param game_state: The dictionary that describes everything on the board.
    :return: A unique key string generated from game state features.
    """
    if game_state is None:
        return None

    # Extract relevant features from the game state
    state = {
        'player_position': game_state['self'][3],  # Assuming game_state['self'] is like ('name', 'score', bomb, (x, y))
        'nearest_enemy_distance': get_nearest_enemy_distance(game_state),  # You need to implement this function
        'nearest_bomb_distance': get_nearest_bomb_distance(game_state),    # You need to implement this function
        'is_in_danger': is_in_danger(game_state),                         # You need to implement this function
        'can_place_bomb': game_state['self'][2],                          # Assuming this indicates bomb availability
        'escape_route_available': is_escape_route_available(game_state),  # You need to implement this function
        'nearest_coin_distance': get_nearest_coin_distance(game_state),   # You need to implement this function
        'coins_remaining': len(game_state['coins'])                       # Assuming game_state['coins'] is a list of coin positions
    }

    return get_state_key(state)

def is_escape_route_available(game_state):
    # This is a placeholder function and should be implemented based on your pathfinding logic
    # The function should return True if there is a safe path the player can take to escape danger
    # For now, let's assume that if the player is not in danger, there's an escape route.
    return not is_in_danger(game_state)

def is_in_danger(game_state):
    player_position = game_state['self'][3]
    for bomb in game_state['bombs']:
        bomb_position, countdown = bomb
        if (player_position[0] == bomb_position[0] and abs(player_position[1] - bomb_position[1]) <= 3) or \
           (player_position[1] == bomb_position[1] and abs(player_position[0] - bomb_position[0]) <= 3):
            return True
    return False

def get_nearest_bomb_distance(game_state):
    player_position = game_state['self'][3]
    bombs = [bomb[0] for bomb in game_state['bombs']]  # Assuming bombs is a list of tuples (position, countdown)
    if not bombs:
        return float('inf')  # If there are no bombs, return a very large distance

    distances = [abs(player_position[0] - bomb[0]) + abs(player_position[1] - bomb[1]) for bomb in bombs]
    return min(distances)

def get_nearest_coin_distance(game_state):
    player_position = game_state['self'][3]
    coins = game_state['coins']
    if not coins:
        return float('inf')  # If there are no coins, return a very large distance

    distances = [abs(player_position[0] - coin[0]) + abs(player_position[1] - coin[1]) for coin in coins]
    return min(distances)

def get_nearest_enemy_distance(game_state):
    player_position = game_state['self'][3]
    enemies = [enemy[3] for enemy in game_state['others']]
    if not enemies:
        return float('inf')  # If there are no enemies, return a very large distance

    distances = [abs(player_position[0] - enemy[0]) + abs(player_position[1] - enemy[1]) for enemy in enemies]
    return min(distances)


def get_state_key(state):
    """
    Generates a unique string key representing the game state.

    :param state: A dictionary containing various features of the game state.
    :return: A string that uniquely represents the given state.
    """
    return (f"{state['player_position'][0]}{state['player_position'][1]}"
            f"{state['nearest_enemy_distance']}"
            f"{state['nearest_bomb_distance']}"
            f"{int(state['is_in_danger'])}"
            f"{int(state['can_place_bomb'])}"
            f"{int(state['escape_route_available'])}"
            f"{state['nearest_coin_distance']}"
            f"{state['coins_remaining']}")