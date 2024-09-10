import os
import pickle
import random

import numpy as np

import glob
import torch
from agent_code.my_nn_agent.model.pytorch_model_alphazero import AlphaZeroNet

if "agent_code" not in os.getcwd():
    os.chdir("agent_code/my_nn_agent")

# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
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
    model_id = 1
    device = torch.device("cpu")
    model_path = f"../../data/model/model_sm1.pt"
    self.model = AlphaZeroNet("MyModel", device, planes=64)
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()
    padding_size = 3
    self.action_prob_move_hist = []

def softmax_on_action_probs(out_policy):
    temperature = 0.9
    action_probs = out_policy.squeeze().numpy()
    action_probs = np.array(action_probs, dtype=np.float32)
    scaled_probs = action_probs / temperature
    exp_probs = np.exp(scaled_probs - np.max(scaled_probs))  # subtract max to prevent overflow
    softmax_probs = exp_probs / np.sum(exp_probs)
    return softmax_probs

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    feature_vector = state_to_features(game_state)


    with torch.no_grad():  # Disable gradient calculation for validation
        states = torch.tensor(np.array(feature_vector), dtype=torch.float32, device=self.model.device)
        out_policy, out_value = self.model(states.unsqueeze(0))
    # action_id = np.argmax(out_policy.numpy())
    use_softmax = False
    if use_softmax:
        action_probs = softmax_on_action_probs(out_policy)
        action_id = np.random.choice(6, p=action_probs)

    else:
        out_policy = out_policy.squeeze().numpy()
        out_policy_copy = out_policy + 0
        values_to_mask = [y for x,y in self.action_prob_move_hist if np.all(x == out_policy)]
        if values_to_mask:
            out_policy[values_to_mask] = -1000
        action_id = np.argmax(out_policy)
        self.action_prob_move_hist.append((out_policy_copy, action_id))
        if len(self.action_prob_move_hist) > 12:
            self.action_prob_move_hist.pop(0)

    print(f"{out_policy} ->  {ACTIONS[action_id]}")
    return ACTIONS[action_id]




def state_to_features(state: dict) -> np.array:
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

    # Encoding various aspects of the state
    encoded_crates = np.array(state["field"] == 1, dtype=np.uint8)
    encoded_walls = np.array(state["field"] == -1, dtype=np.uint8)

    # Encode the positions of other agents and whether they can bomb
    encoded_other_agents = np.zeros((17, 17), dtype=np.uint8)
    encoded_other_agents_can_bomb = np.zeros((17, 17), dtype=np.uint8)
    for _, _, can_bomb, (row, col) in state["others"]:
        encoded_other_agents[row, col] = 1
        if can_bomb:
            encoded_other_agents_can_bomb[row, col] = 1

    # Encode whether the player can place a bomb
    encoded_self_can_bomb = np.ones((17, 17), dtype=np.uint8) if state["self"][2] else np.zeros((17, 17),
                                                                                                dtype=np.uint8)

    # Encode bomb positions and their explosion timers
    encoded_bombs = np.zeros((4, 17, 17), dtype=np.uint8)
    for (bomb_row, bomb_col), countdown in state["bombs"]:
        encoded_bombs[countdown][bomb_row][bomb_col] = 1
        for d_row, d_col in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            for distance in range(1, 4):
                r_offset, c_offset = d_row * distance, d_col * distance
                if state["field"][bomb_row + r_offset][bomb_col + c_offset] == -1:
                    break
                encoded_bombs[countdown][bomb_row + r_offset][bomb_col + c_offset] = 1

    # Merge explosion map into the bombs encoding
    encoded_explosions = np.array(state["explosion_map"], dtype=np.uint8)
    encoded_bombs[0] = encoded_explosions | encoded_bombs[0]

    # Encode the positions of coins
    encoded_coins = np.zeros((17, 17), dtype=np.uint8)
    for coin_row, coin_col in state["coins"]:
        encoded_coins[coin_row, coin_col] = 1


    combined_encoded_state = np.stack((encoded_walls, encoded_crates, encoded_self_can_bomb, encoded_coins,
                                       encoded_other_agents, encoded_other_agents_can_bomb), axis=0)
    combined_encoded_state = np.concatenate((combined_encoded_state, encoded_bombs), axis=0).astype(np.float32)

    # Step 4: Apply padding to create a framed view around the player
    padding_size = 3  # Equivalent to a 9x9 window with player centered. Note there is already a frame with the width 1

    padded_encoded_state = np.empty((combined_encoded_state.shape[0],
                                     combined_encoded_state.shape[1] + (padding_size) * 2,
                                     combined_encoded_state.shape[2] + (padding_size) * 2), dtype=np.float32)
    padded_encoded_state[0] = np.pad(combined_encoded_state[0], pad_width=padding_size, mode='constant',
                                     constant_values=1)
    padded_encoded_state[1:] = np.pad(combined_encoded_state[1:],
                                      pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                                      mode='constant', constant_values=0)


    # Step 5: Extract the framed view around the player
    player_row, player_col = [coord + padding_size for coord in state["self"][3]]
    final_encoded_state = padded_encoded_state[
                          :,
                          player_row - padding_size - 1:player_row + padding_size + 1 + 1,
                          player_col - padding_size - 1:player_col + padding_size + 1 + 1
                          ]

    return final_encoded_state

