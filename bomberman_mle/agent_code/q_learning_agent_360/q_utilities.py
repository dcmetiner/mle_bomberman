import glob
import pickle
import numpy as np
import random
from datetime import datetime
import os
import re
from collections import deque

def canonicalize_neighbours(neighbour_tiles):
    """Normalize the neighbor tiles to account for symmetrical/mirrored states."""
    # Mirror horizontally and vertically (rotating could also be used)
    smallest_neighbour_tiles_representation = neighbour_tiles
    neighbour_tiles_to_rotate = neighbour_tiles
    rotate_times = 0
    for rotation_cnt in range(1, 4):
        neighbour_tiles_to_rotate = neighbour_tiles_to_rotate[1:4] + neighbour_tiles_to_rotate[0:1]
        if smallest_neighbour_tiles_representation > neighbour_tiles_to_rotate:
            smallest_neighbour_tiles_representation = neighbour_tiles_to_rotate
            rotate_times = rotation_cnt
    # Return the lexicographically smallest state
    return smallest_neighbour_tiles_representation, rotate_times


def get_normalized_neighbour_tiles_to_id_dict():
    all_neighbour_tiles = []
    state_size = 3
    for up in range(state_size):
        for left in range(state_size):
            for down in range(state_size):
                for right in range(state_size):
                    all_neighbour_tiles.append((up, left, right, down))
    all_normalized_neighbour_tiles = []
    for neighbour_tiles in all_neighbour_tiles:
        normalized_neighbour_tiles, _ = canonicalize_neighbours(neighbour_tiles)
        if normalized_neighbour_tiles not in all_normalized_neighbour_tiles:
            all_normalized_neighbour_tiles.append(normalized_neighbour_tiles)
    normalized_neighbour_tiles_to_id = {tile: index for index, tile in enumerate(all_normalized_neighbour_tiles)}
    return normalized_neighbour_tiles_to_id

# kill_events = glob.glob("data/pure_state_data/*kill_events*.pkl")[0]
# with open(kill_events, "rb") as file:
#     kill_events = pickle.load(file)

DIRECTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # UP, LEFT, DOWN, RIGHT
NEIGHBOURS_STATE_TO_ID = get_normalized_neighbour_tiles_to_id_dict()
ID_TO_NEIGHBOURS_STATE = {val:key for key, val in NEIGHBOURS_STATE_TO_ID.items()}

ACTION_ROTATION_BACKWARD = {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}


def get_actual_move_before_rotation(feature_vector, action):
    if action in ['BOMB', 'WAIT']:
        return action
    neighbour_tiles = tuple(feature_vector[:4])
    _, rotation_count = canonicalize_neighbours(neighbour_tiles)
    for _ in range(rotation_count):
        action = ACTION_ROTATION_BACKWARD[action]
    return action


def rotate_back_n_times(action, rotation_count):
    if action in ['BOMB', 'WAIT']:
        return action
    for _ in range(rotation_count):
        action = ACTION_ROTATION_BACKWARD[action]
    return action


def print_state(state, mirrowed=True, return_field=False, logger=None):
    # ANSI color codes
    RESET_COLOR = "\033[0m"
    RED_COLOR = "\033[91m"
    BLUE_COLOR = "\033[94m"
    GREEN_COLOR = "\033[92m"
    YELLOW_COLOR = "\033[93m"

    field = state["field"].tolist()

    for x in range(17):
        for y in range(17):
            if field[x][y] == -1:
                field[x][y] = "x"
            elif field[x][y] == 1:
                field[x][y] = "-"
            elif field[x][y] == 0:
                field[x][y] = " "
            if state["explosion_map"][x, y] == 1:
                field[x][y] = "E"

    self_x, self_y = state["self"][3]
    field[self_x][self_y] = f"{RED_COLOR}s{RESET_COLOR}"  # Self is red

    for _, _, _, (x, y) in state["others"]:
        field[x][y] = f"{BLUE_COLOR}g{RESET_COLOR}"  # Others are blue

    for (x, y) in state["coins"]:
        field[x][y] = f"{GREEN_COLOR}C{RESET_COLOR}"  # Others are blue

    for (x, y), countdown in state["bombs"]:
        if field[x][y] == f"{RED_COLOR}s{RESET_COLOR}":
            field[x][y] = f"{RED_COLOR}S{RESET_COLOR}"
        elif field[x][y] == f"{BLUE_COLOR}g{RESET_COLOR}":
            field[x][y] = f"{BLUE_COLOR}G{RESET_COLOR}"
        else:
            field[x][y] = countdown

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


def can_escape(state, pos, dangerous_fields, others_pos, bombs_pos, step_taken=True):
    start_x, start_y = pos
    queue = deque([[(start_x, start_y), 1 if step_taken else 0]])
    while queue:
        (x, y), step = queue.popleft()
        if np.any(dangerous_fields[step - 1: min(step + 1, len(dangerous_fields)), x, y] == 1):
            continue

        if not np.any(dangerous_fields[step + 1:, x, y] == 1):
            return True

        for dx, dy in DIRECTIONS:
            new_x, new_y = x + dx, y + dy
            if is_position_moveable(state, (new_x, new_y), (others_pos+bombs_pos)):
                queue.append([(new_x, new_y), step + 1])
            queue.append([(x, y), step + 1])
    return False


def is_new_location_closer_to_target(current_pos, next_pos, state, game_mode):
    target_positions = find_target_positions(state, game_mode)



def get_dangerous_fields(state):
    dangerous_fields = np.zeros((5, 17, 17), dtype=np.uint8)
    for (bomb_x, bomb_y), countdown in state["bombs"]:
        dangerous_fields[countdown + 1][bomb_x, bomb_y] = 1
        for (x_dir, y_dir) in DIRECTIONS:
            for distance in range(1, 4):
                x_offset, y_offset = x_dir * distance, y_dir * distance
                dang_x, dang_y = bomb_x + x_offset, bomb_y + y_offset
                if state["field"][dang_x, dang_y] == -1:
                    break
                dangerous_fields[countdown + 1][dang_x, dang_y] = 1
    dangerous_fields[0] = state["explosion_map"].astype(np.uint8)
    return dangerous_fields


def get_game_mode(state, all_coins_collected=False):
    game_mode = None
    if state["coins"]:
        game_mode = 0
    elif not np.any(state["field"] == 1) or all_coins_collected:
        game_mode = 2
    else:
        game_mode = 1
    assert game_mode in [0, 1, 2]
    return game_mode


def is_position_moveable(state, position, others_pos):
    return state["field"][position] == 0 and position not in others_pos


def get_path_length_to_target(state, start, dangerous_fields, target_pos):
    path =  len(state, start, dangerous_fields, target_pos)
    if path:
        return len(path)

def bfs_shortest_path(state, start, dangerous_fields, target_pos, step_taken=True):
    queue = deque([(start, [])])  # Queue holds tuples of (current position, path)
    visited = set([start])
    if start in target_pos:
        return [start], int(step_taken)
    if target_pos:
        while queue:
            (x, y), path = queue.popleft()
            # print(f"Popped pos: {(x,y)} and path: {path} from queue")
            index = next((index for index, (i, j) in enumerate(target_pos) if abs(x - i) + abs(y - j) == 1), None)
            if index is not None:
                return path + [(x, y), target_pos[index]], len(path) + step_taken + 1
            if len(path) > 20:
                break
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                step = len(path) + step_taken
                if state["field"][nx, ny] == 0 and (nx, ny) not in visited and not np.any(
                        dangerous_fields[step: min(step + 1, len(dangerous_fields)), x, y] == 1):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(x, y)]))
                    # print(f"updated visited is: {visited} and new queue is: {queue}")

    return None, None


def get_closeness_state(state, dangerous_fields, target_positions, others_pos, step_taken=False):
    self_pos = state["self"][3]
    if len(target_positions) == 0:
        return 0
    path_to_target, move_cnt_to_target = bfs_shortest_path(state, self_pos, dangerous_fields, target_positions, step_taken)
    target = path_to_target[-1]
    min_path_len_for_others = None
    for other_x, other_y in others_pos:
        if abs(target[0] - other_x) + abs(target[1] - other_y) > move_cnt_to_target:
            continue
        move_cnt_to_target_for_other = bfs_shortest_path(state, (other_x, other_y), dangerous_fields, target_positions, step_taken)[1]
        if min_path_len_for_others is None or (min_path_len_for_others is not None and min_path_len_for_others > move_cnt_to_target_for_other):
            min_path_len_for_others = move_cnt_to_target_for_other
    if move_cnt_to_target <= 5:
        if move_cnt_to_target < min_path_len_for_others:
            return 1
        else:
            return 2
    elif move_cnt_to_target <= 10:
        if move_cnt_to_target < min_path_len_for_others:
            return 3
        else:
            return 4
    return 0


def get_neightbour_tile_states(state, game_mode, dangerous_fields, others_pos, bombs_pos):
    # print_state(state)

    neighbour_states = [0] * 4
    self_x, self_y = state["self"][3]
    for id, (x_dir, y_dir) in enumerate(DIRECTIONS):
        pos_x, pos_y = self_x + x_dir, self_y + y_dir
        if state["field"][pos_x, pos_y] != 0 or (pos_x, pos_y) in (others_pos+bombs_pos) or not can_escape(state, (pos_x, pos_y), dangerous_fields, others_pos, bombs_pos):
            neighbour_states[id] = 2
        else:
            neighbour_states[id] = 0
    target_positions = find_target_positions(state, game_mode)
    # if game_mode == 0:
    #     print(f"target_positions: {target_positions}")
    path, move_cnt_to_target = bfs_shortest_path(state, (self_x, self_y), dangerous_fields, target_positions)
    # print(f"Path length to target at (current) position {(self_x, self_y)}: {move_cnt_to_target} and path: {path}")

    if move_cnt_to_target:
        move_cnt_from_neighbours_to_target = [None] * 4
        shortest_path_length = move_cnt_to_target
        for id, (x_dir, y_dir) in enumerate(DIRECTIONS):
            if neighbour_states[id] == 2:
                continue
            pos_x, pos_y = self_x + x_dir, self_y + y_dir
            path, move_cnt = bfs_shortest_path(state, (pos_x, pos_y), dangerous_fields, target_positions)
            # print(f"Path length to target at position {(pos_x, pos_y)}: {move_cnt} and path: {path}")
            move_cnt_from_neighbours_to_target[id] = move_cnt
            # print("move_cnt: ", move_cnt, ", shortest_path_length: ", shortest_path_length )
            if move_cnt and move_cnt < shortest_path_length:
                shortest_path_length = move_cnt
        if shortest_path_length < move_cnt_to_target:
            for id, path_len in enumerate(move_cnt_from_neighbours_to_target):
                if path_len is not None and path_len == shortest_path_length:
                    neighbour_states[id] = 1
    # print(neighbour_states)

    # print("****************************")
    return neighbour_states

def find_target_positions(state, game_mode):
    if game_mode == 0:
        return state["coins"]
    elif game_mode == 1:
        crate_locations = np.argwhere(state["field"] == 1)
        return [tuple(loc) for loc in crate_locations]
    elif game_mode == 2:
        return [other[3] for other in state["others"]]


def get_bomb_effect(state, others_pos):
    self_x, self_y = state["self"][3]
    bomb_effect = 0
    bomb_map = np.zeros((17,17), dtype=np.uint8)
    bomb_map[self_x, self_y] = 1
    for (x_dir, y_dir) in DIRECTIONS:
        for distance in range(1, 4):
            x_offset, y_offset = x_dir * distance, y_dir * distance
            dang_x, dang_y = self_x + x_offset, self_y + y_offset
            if state["field"][dang_x, dang_y] == -1:
                break
            if state["field"][dang_x, dang_y] == 1 or (dang_x, dang_y) in others_pos:
                bomb_effect += 1
            bomb_map[dang_x, dang_y] = 1
    return bomb_effect, bomb_map

def get_encoded_state_on_current_tile(state, dangerous_fields, others_pos, bombs_pos):
    _, _, can_bomb, (self_x, self_y) = state["self"]
    s = None
    bomb_effect, bomb_map = get_bomb_effect(state, others_pos) if can_bomb else (0, None)
    extended_dangerous_fields = np.concatenate((dangerous_fields, bomb_map.reshape((1,17,17))), axis=0) if bomb_effect else dangerous_fields
    if can_escape(state, (self_x, self_y), extended_dangerous_fields, others_pos, bombs_pos):
        if 0 < bomb_effect < 3:
            s = 1
        elif 3 <= bomb_effect <= 5:
            s = 2
        elif 5 < bomb_effect:
            s = 3
    elif not can_escape(state, (self_x, self_y), dangerous_fields, others_pos, bombs_pos):
        s = 4
    if s is None:
        s = 0
    return s



def feature_vector_to_id(feature_vector, return_num_rotations = True):
    assert len(feature_vector) == 6
    neighbour_tiles = tuple(feature_vector[:4])
    current_tile = feature_vector[4]
    game_mode = feature_vector[5]

    # Define the bases for each set of values
    # base_neighbour = 3  # 3 possible values for each neighbour tile
    base_current = 5  # 5 possible values for the current tile
    base_game_mode = 3  # 3 possible values for the game mode
    # base_closeness_state = 5

    # Canonicalize the neighbor tiles by mirroring
    normalized_neighbours, rot_cnt = canonicalize_neighbours(neighbour_tiles)
    neighbours_state_id = NEIGHBOURS_STATE_TO_ID[normalized_neighbours]

    # Combine the state into a unique index using positional encoding
    state_id = (neighbours_state_id * base_current * base_game_mode+
                current_tile * base_game_mode +  # Move by the base of the game mode
                game_mode)

    assert(state_id in range(360))
    if return_num_rotations:
        return state_id, rot_cnt
    return state_id

def state_to_feature_vector(state):
    others_pos = [other[3] for other in state["others"]]
    bombs_pos = [pos for pos, countdown in state["bombs"]]

    game_mode = get_game_mode(state)

    dangerous_fields = get_dangerous_fields(state)
    # dangerous_fields.sum(axis=(1, 2))

    neighbour_tiles = get_neightbour_tile_states(state, game_mode, dangerous_fields, others_pos, bombs_pos)

    current_tile = get_encoded_state_on_current_tile(state, dangerous_fields, others_pos, bombs_pos)

    # target_positions = find_target_positions(state, game_mode)
    # closeness_state = get_closeness_state(state, dangerous_fields, target_positions, others_pos, step_taken=False)

    # encoded_state = neighbour_tiles + [current_tile] + [game_mode] + [closeness_state]

    encoded_state = neighbour_tiles + [current_tile] + [game_mode]

    assert len(encoded_state) == 6

    return encoded_state



def state_to_id(state, return_num_rotations = True):
    others_pos = [other[3] for other in state["others"]]
    bombs_pos = [pos for pos, countdown in state["bombs"]]

    # game_mode = get_game_mode(state)
    #
    # dangerous_fields = get_dangerous_fields(state)
    # # dangerous_fields.sum(axis=(1, 2))
    #
    # neighbour_tiles = get_neightbour_tile_states(state, game_mode, dangerous_fields, others_pos, bombs_pos)
    #
    # current_tile = get_encoded_state_on_current_tile(state, dangerous_fields, others_pos, bombs_pos)
    #
    # encoded_state = neighbour_tiles + [current_tile] + [game_mode]

    feature_vector = state_to_feature_vector(state)
    assert len(feature_vector) == 6

    return feature_vector_to_id(feature_vector, return_num_rotations)





