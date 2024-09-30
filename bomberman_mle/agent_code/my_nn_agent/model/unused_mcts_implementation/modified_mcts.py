import math
import numpy as np
import glob
import torch
import pickle
import re

DIRECTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # UP, LEFT, DOWN, RIGHT

def get_cropped_state(combined_encoded_state, player_pos):
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
    player_row, player_col = [coord + padding_size for coord in player_pos]
    final_encoded_state = padded_encoded_state[
                          :,
                          player_row - padding_size - 1:player_row + padding_size + 1 + 1,
                          player_col - padding_size - 1:player_col + padding_size + 1 + 1
                          ]

    return final_encoded_state

def get_encoded_state(state: dict) -> np.array:
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

    return combined_encoded_state

def get_final_state_for_network(state):
    player_pos = state["self"][3]

    encoded_state = get_encoded_state(state)

    final_state = get_cropped_state(encoded_state, player_pos)

    return final_state


def get_valid_moves(feature):
    valid_move_ids = [0] * 6
    c = feature[0].shape[1] // 2  # center
    is_occupied_big = np.any(feature[[0, 1, 4, 6]], axis=0).astype(int)
    is_occupied = is_occupied_big[c - 1:c + 2, c - 1:c + 2]

    if is_occupied[1, 0] == 0:
        valid_move_ids[0] = 1
    if is_occupied[0, 1] == 0:
        valid_move_ids[1] = 1
    if is_occupied[1, 2] == 0:
        valid_move_ids[2] = 1
    if is_occupied[2, 1] == 0:
        valid_move_ids[3] = 1
    if feature[2, 0, 0] == 1:
        valid_move_ids[4] = 1
    if is_occupied[1, 1] == 0:
        valid_move_ids[5] = 1
    return valid_move_ids


def check_if_self_died(node):
    feature_vector = get_cropped_state(node.encoded_state, node.player_info[3])

    return np.any(get_valid_moves(feature_vector))


def get_is_terminal_value(node):
    if check_if_self_died(node):
        return (True, -1)
    if check_if_self_scored(node) and not check_if_self_in_danger(node):
        return (True, 1)
    return False, 0



def check_if_self_scored(node):
        if node.parent:
            score = node.player_info[1]
            parent_score = node.parent.player_info[1]
            return score > parent_score
        return False


def check_if_self_in_danger(node):
    feature_vector = get_cropped_state(node.encoded_state, node.player_info[3])
    c = feature_vector[0].shape[1] // 2  # center
    is_occupied_big = np.any(feature_vector[6:], axis=0).astype(int)
    is_occupied = is_occupied_big[c - 1:c + 2, c - 1:c + 2]
    x,y = node.player_info[3]
    return np.any(is_occupied[:,x,y])



def print_encoded_state(encoded_state, player_info, mirrowed=True, return_field=False, logger=None):
    # ANSI color codes
    RESET_COLOR = "\033[0m"
    RED_COLOR = "\033[91m"
    BLUE_COLOR = "\033[94m"
    GREEN_COLOR = "\033[92m"
    YELLOW_COLOR = "\033[93m"

    field = [[" " for i in range(17)] for j in range(17)]

    for x in range(17):
        for y in range(17):
            if encoded_state[0][x][y]:
                field[x][y] = "x"
            elif encoded_state[1][x][y]:
                field[x][y] = "-"
            else:
                field[x][y] = " "

    self_x, self_y = player_info[3]
    field[self_x][self_y] = f"{RED_COLOR}s{RESET_COLOR}"  # Self is red

    others_x, others_y = np.where(encoded_state[4] == 1)
    for i in range(len(others_x)):
        field[others_x[i]][others_y[i]] = f"{BLUE_COLOR}g{RESET_COLOR}"  # Others are blue

    coins_x, coins_y = np.where(encoded_state[3] == 1)
    for i in range(len(coins_x)):
        field[coins_x[i]][coins_y[i]] = f"{GREEN_COLOR}C{RESET_COLOR}"


    for index, bomb_matrix in enumerate(encoded_state[6:]):
        bomb_x, bomb_y = np.where(encoded_state[6+index] == 1)
        for i in range(len(bomb_x)):
            field[bomb_x[i]][bomb_y[i]] = f"{GREEN_COLOR}C{RESET_COLOR}"

            if field[bomb_x[i]][bomb_y[i]] == f"{RED_COLOR}s{RESET_COLOR}":
                field[bomb_x[i]][bomb_y[i]] = f"{RED_COLOR}S{RESET_COLOR}"
            elif field[bomb_x[i]][bomb_y[i]] == f"{BLUE_COLOR}g{RESET_COLOR}":
                field[bomb_x[i]][bomb_y[i]] = f"{BLUE_COLOR}G{RESET_COLOR}"
            else:
                field[bomb_x[i]][bomb_y[i]] = index



    # Printing the field
    if logger:
        field = remove_color_codes(field)
        for x in range(17):
            row_output = []
            for y in range(17):
                if mirrowed:
                    row_output.append(str(field[y][x]))
                else:
                    row_output.append(str(field[x][y]))

            logger(" ".join(row_output))
    else:
        for x in range(17):
            for y in range(17):
                if mirrowed:
                    print(str(field[y][x]), end=" ")
                else:
                    print(str(field[x][y]), end=" ")

            print()
    if return_field:
        return field


def remove_color_codes(field):
    """Remove ANSI color codes from each element in the field."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    # Iterate through the field and remove the color codes from each element
    for x in range(len(field)):
        for y in range(len(field[0])):
            if not isinstance(field[x][y], int):
                field[x][y] = ansi_escape.sub('', field[x][y])

    return field

class Node:
    def __init__(self, args, encoded_state, player_info, parent=None, action_taken=None, prior=0, visit_count=0,  bomb_map_self=None):
        self.args = args
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0
        self.encoded_state = encoded_state
        self.player_info = player_info
        if bomb_map_self is None:
            self.bomb_map_self = np.zeros((4,17,17), dtype=np.float32)
        else:
            self.bomb_map_self = bomb_map_self

    def is_leaf_node(self):
        return len(self.children) == 0

    def get_child_puct(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

class ModifiedMonteCarloTreeSearch:
    def __init__(self, args:dict, model):
        self.args = args
        self.model = model

    def expand(self, node, policy, valid_moves):

        for move_action_id in range(6):

            if valid_moves[move_action_id]:
                child_encoded_state, child_player_info, child_bomb_map = self.make_move(node, move_action_id)
                child = Node(self.args, encoded_state=child_encoded_state, player_info=child_player_info, parent=node,
                             action_taken=move_action_id, prior=policy[move_action_id], visit_count=0,
                             bomb_map_self=child_bomb_map)

                node.children.append(child)


    def make_move(self,node, move_id):
        encoded_state = node.encoded_state
        player_info = node.player_info
        child_bomb_map_self = node.bomb_map_self.copy()

        child_state = encoded_state.copy()

        child_can_bomb = player_info[2]
        child_score = player_info[1]

        bomb_map_from_current_step = np.zeros_like(child_bomb_map_self[0])

        child_player_info = None

        if np.any(child_bomb_map_self[0] == 1):
            for x in range(child_bomb_map_self[0].shape[0]):
                for y in range(child_bomb_map_self[0].shape[1]):
                    if child_bomb_map_self[0, x, y] == 1 and encoded_state[4, x, y] == 1:
                        encoded_state[4, x, y] = 0
                        encoded_state[5, x, y] = 0
                        child_score += 5

        explosion_map = encoded_state[6]
        destoyed_crate_count = 0
        if np.any(explosion_map == 1):
            for x in range(explosion_map.shape[0]):
                for y in range(explosion_map.shape[1]):
                    if explosion_map[x, y] == 1 and encoded_state[1, x, y] == 1:
                        encoded_state[1, x, y] = 0
                        destoyed_crate_count += 1

        if np.all(child_bomb_map_self == 0):
            child_can_bomb = True
        else:
            child_can_bomb = False

        if 0 <= move_id < 4 or move_id == 5:
            player_pos = player_info[3]
            if move_id == 5:
                x, y = 0, 0
            else:
                x, y = DIRECTIONS[move_id]
            nx, ny = player_pos[0] + x, player_pos[1] + y
            if child_state[3, nx, ny] == 1:
                child_score += 1
                child_state[3, nx, ny] = 0
            child_player_info = tuple((player_info[0] , child_score, child_can_bomb, (nx,ny)))

        if move_id == 4:
            bomb_row, bomb_col = player_info[3]
            bomb_map_from_current_step[bomb_row][bomb_col] = 1
            for d_row, d_col in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                for distance in range(1, 4):
                    r_offset, c_offset = d_row * distance, d_col * distance
                    if encoded_state[0][bomb_row, bomb_col] == 1:
                        break
                    bomb_map_from_current_step[bomb_row, bomb_col] = 1
        return child_state, child_player_info, child_bomb_map_self


    def backpropagate(self,node, value, player_self=True):  # TODO: Player self
        """
        Backpropagates the value of a leaf node back up to its ancestors.

        :param node: The node from which the backpropagation begins.
        :param value: The value to be backpropagated.
        :param is_white_player_terminated: Boolean flag indicating whether the white player is terminated.
        """

        node.value_sum += value
        node.visit_count += 1
        if node.parent is not None:
            self.backpropagate(node.parent, value)

    def select_leaf_node(self, node: Node):
        """
        Selects the leaf node from the current node for expansion.

        :param node: The node from which the leaf node selection starts.
        :return: The selected leaf node.
        """
        if node.is_leaf_node():
            return node
        best_child = None
        best_puct = -np.inf
        for child in node.children:
            puct = node.get_child_puct(child)
            if puct > best_puct:
                best_child = child
                best_puct = puct
        return self.select_leaf_node(best_child)

    def apply_valid_move_mask_to_policy(self, policy, valid_moves, add_noise=False):
        mask = np.array(valid_moves)
        policy *= mask
        policy /= policy.sum()
        if add_noise:
            dirichlet_noise = np.random.dirichlet([self.args['dirichlet_alpha']] * mask.sum())
            policy[np.where(mask == 1)] = (1 - self.args['dirichlet_epsilon']) * policy[np.where(mask == 1)] + self.args[
                'dirichlet_epsilon'] * dirichlet_noise
        return policy


    def search(self, encoded_state, player_info, bomb_map_self):


        root = Node(self.args, encoded_state, player_info, parent=None, action_taken=None, prior=0, visit_count=0)
        feature_vector = get_cropped_state(root.encoded_state, player_info[3])
        with torch.no_grad():
            policy, _ = self.model(torch.tensor(feature_vector, device=self.model.device).unsqueeze(0))
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        valid_moves = get_valid_moves(feature_vector)

        policy = self.apply_valid_move_mask_to_policy(policy, valid_moves, add_noise=False)
        self.expand(root, policy, valid_moves)
        self.backpropagate(root, 0, True)

        for search in range(self.args['num_searches']):
            node = self.select_leaf_node(root)
            is_terminal, value = get_is_terminal_value(node)
            if not is_terminal:
                feature_vector = get_cropped_state(node.encoded_state, node.player_info[3])
                with torch.no_grad():
                    policy, _ = self.model(torch.tensor(feature_vector, device=self.model.device).unsqueeze(0))
                    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = get_valid_moves(feature_vector)
                policy = self.apply_valid_move_mask_to_policy(policy, valid_moves, add_noise=False)
                value = value.item()
                self.expand(node, policy, valid_moves)
            self.backpropagate(node, value)

        action_probs_all = np.zeros(6)
        valid_move_idx = [child.action_taken for child in root.children]
        action_probs_all[valid_move_idx] = [child.visit_count for child in root.children]
        # action_probs_all /= np.sum(action_probs_all)
        return action_probs_all


def get_inputs_for_mcts_from_state(state, bomb_map_self = None):
    encoded_state = get_encoded_state(state)
    player_info = state["self"]
    if bomb_map_self is None:
        bomb_map_self = np.zeros((4, 17, 17), dtype=np.float32)
    else:
        bomb_map_self = bomb_map_self
    return encoded_state, player_info, bomb_map_self





sample_args = {
    "terminate_cnt": 225,
    'C': 4,
    'num_searches': 250, #  In this project, this variable was always '250' or more. Use this variable with a low value only for testing purposes
    'num_iterations': 1000,
    'num_selfPlay_iterations': 10,
    "num_evaluation_iterations": 10,
    "num_parallel_games": 10,
    'num_epochs': 100,
    'batch_size': 256,
    'temperature': 1,
    'temperature_threshold': 25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1,
    'patience': 3,
    'winning_threshold': 0.55
}


