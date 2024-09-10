# TODO: This script is going to be improved


prepare_training_data = False
if prepare_training_data:

    import glob
    import pickle
    import numpy as np
    import random
    from agent_code.my_nn_agent.model.pytorch_model_alphazero import AlphaZeroNet
    from datetime import datetime
    import os

    coin_collection_training_data = []
    kill_events_training_data = []
    death_events_training_data = []

    # Step 1: Load and sort game state files
    state_files = glob.glob("data/*states*")
    state_files.sort()
    for file_name in state_files:
        if os.path.getsize(file_name) == 0:
            print(f"{file_name} is empty")
            continue
        # Step 2: Load initial game state data from the first file
        try:
            with open(file_name, 'rb') as file:
                game_data = pickle.load(file)
        except EOFError:
            print(f"{file_name} is empty or corrupted. Please check the file content.")
            continue

        # Restricting data to the first 100 entries for demonstration
        # game_data = game_data[:2000]

        # Step 3: Map player names to player IDs
        playername_to_id = {}
        # new_game_idx = 0
        # for idx, state in enumerate(game_data):
        #     if state["step"]  == 1:
        #         new_game_idx = idx
        #         break
        new_game_idx = next((idx for idx, state in enumerate(game_data) if state["step"] == 1), 0)
        for idx in range(4):
            player_name = game_data[new_game_idx+idx]["self"][0]
            playername_to_id[player_name] = idx

        # Step 4: Organize game data by player and round
        player_data_by_round = [[[]] for _ in range(4)]
        current_round = game_data[0]["round"]

        idx = 0
        while idx < len(game_data):
            player_id = playername_to_id[game_data[idx]["self"][0]]
            player_data_by_round[player_id][-1].append(game_data[idx])

            idx += 1

            # Check if the next state is in a new round
            if idx < len(game_data) and game_data[idx]["round"] != current_round:
                current_round = game_data[idx]["round"]
                for player_data in player_data_by_round:
                    player_data.append([])

        # Step 5: Flatten data structure to a single list of all game states
        # all_game_data = [round_data for player_data in player_data_by_round for round_data in player_data]
        all_game_data = [round_data for player_data in player_data_by_round for round_data in player_data[:-1]]  # TODO




        wrong_moves = 0
        total_moves = 0
        for round_data in all_game_data[:4]:
            for i in range(len(round_data) - 1):
                x, y = round_data[i]["self"][3]
                x_next, y_next = round_data[i + 1]["self"][3]
                if x == x_next and y == y_next:
                    act = "BOMB" if round_data[i]["self"][2] == True and round_data[i + 1]["self"][
                        2] == False else "WAIT"
                elif x == x_next:
                    if y < y_next:
                        act = "DOWN"
                    else:
                        act = "UP"
                elif y == y_next:
                    if x < x_next:
                        act = "RIGHT"
                    else:
                        act = "LEFT"
                total_moves +=1
                if act != round_data[i]["act"]:
                    wrong_moves += 1
        print("***********************")
        print(f"File Name: {file_name} \n-> There are {str(wrong_moves)} wrong moves in {str(total_moves)} total moves")



        # Verify length of data
        print(f"Total game rounds collected: {len(all_game_data)}")
        # print(f"Length of the first game round data: {len(all_game_data[0])}") # thic can fail if the first player has already been dead

        # Step 6: Initialize lists to collect specific events
        coin_collection_events = []
        kill_events = []
        death_events = []

        # Step 7: Collect events based on score changes
        for game_round in all_game_data:
            previous_score = 0

            for state_id, state in enumerate(game_round):
                current_score = state['self'][1]

                if current_score == previous_score:
                    continue
                elif current_score > previous_score:
                    if current_score == previous_score + 1:
                        # Collect coin event
                        relevant_history = game_round[max(0, state_id - 7): state_id]
                        coin_collection_events.append(relevant_history)
                    else:
                        # Kill event (score increased by more than 1)
                        relevant_history = game_round[max(0, state_id - 7): state_id]
                        kill_events.append(relevant_history)
                    previous_score = current_score

            # Check if the game ended early (agent death)
            if len(game_round) > 0 and game_round[-1]['step'] < 400 and len(game_round[-1]['others']) != 0:
                relevant_history = game_round[max(0, len(game_round) - 7): len(game_round)]
                death_events.append(relevant_history)

        # Output the lengths of collected event data
        print(f"Coin collection events: {len(coin_collection_events)}")
        print(f"Kill events: {len(kill_events)}")
        print(f"Death events: {len(death_events)}")

        coin_collection_events = [state for state_list in coin_collection_events for state in state_list]
        kill_events = [state for state_list in kill_events for state in state_list]
        death_events = [state for state_list in death_events for state in state_list]

        # random.shuffle(coin_collection_events)
        # random.shuffle(kill_events)
        # random.shuffle(death_events)

        limit = max(len(kill_events), len(death_events)) + 99999999
        # Step 1: Define the collection with reward and limits for events
        event_collections_with_rewards = [
            (coin_collection_events, 0.7, limit),
            (kill_events, 1, limit),
            (death_events, -1, limit)
        ]

        action_to_index = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3, "BOMB": 4, "WAIT": 5}  # TODO

        # Step 2: Process each event collection
        for event_collection, reward_value, limit in event_collections_with_rewards:
            encoded_states = []  # List to store the encoded states for training
            for state in event_collection[:limit]:

                # Encode the action probabilities based on the action taken
                action_probs = np.zeros(6, dtype=np.float32)
                if state["act"] is None:
                    action_probs[action_to_index["WAIT"]] = 1

                else:
                    action_probs[action_to_index[state["act"]]] = 1

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
                encoded_self_can_bomb = np.ones((17, 17), dtype=np.uint8) if state["self"][2] else np.zeros((17, 17), dtype=np.uint8)

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

                # Step 3: Combine all encoded layers into a single state representation
                # combined_encoded_state = np.concatenate((
                #     np.array([
                #         encoded_walls, encoded_crates, encoded_self_can_bomb,
                #         encoded_coins, encoded_other_agents, encoded_other_agents_can_bomb
                #     ], dtype=np.float32),
                #     encoded_bombs
                # ), axis=0)
                # Concatenate all encodings into a single state representation
                combined_encoded_state = np.stack((encoded_walls, encoded_crates, encoded_self_can_bomb, encoded_coins, encoded_other_agents, encoded_other_agents_can_bomb), axis=0)
                combined_encoded_state = np.concatenate((combined_encoded_state, encoded_bombs), axis=0).astype(np.float32)

                # Step 4: Apply padding to create a framed view around the player
                padding_size = 3  # Equivalent to a 9x9 window with player centered. Note there is already a frame with the width 1

                padded_encoded_state = np.empty((combined_encoded_state.shape[0], combined_encoded_state.shape[1]+(padding_size)*2, combined_encoded_state.shape[2]+(padding_size)*2), dtype=np.float32)
                padded_encoded_state[0] = np.pad(combined_encoded_state[0], pad_width=padding_size, mode='constant', constant_values=1)
                padded_encoded_state[1:] = np.pad(combined_encoded_state[1:], pad_width=((0, 0), (padding_size,padding_size), (padding_size,padding_size)), mode='constant', constant_values=0)
                padded_encoded_state[2] = np.ones_like(padded_encoded_state[2]) if state["self"][2] else np.zeros_like(
                    padded_encoded_state[2])

                # padded_encoded_state = np.pad(
                #     combined_encoded_state,
                #     pad_width=((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                #     mode='constant',
                #     constant_values=((0, 0), (1, 0), (1, 0))
                # )

                # Step 5: Extract the framed view around the player
                player_row, player_col = [coord + padding_size for coord in state["self"][3]]
                final_encoded_state = padded_encoded_state[
                    :,
                    player_row - padding_size-1:player_row + padding_size + 1 + 1,
                    player_col - padding_size-1:player_col + padding_size + 1 + 1
                ]

                # Step 6: Store the encoded state with action probabilities and rewards
                encoded_states.append((final_encoded_state, action_probs, reward_value))

            if reward_value == 0.7:
                coin_collection_training_data += encoded_states
            elif reward_value == 1:
                kill_events_training_data += encoded_states
            elif reward_value == -1:
                death_events_training_data += encoded_states




    timestamp = datetime.now(tz=None).strftime("%d-%b-%Y(%H:%M:%S)")
    path = "data/training_data/"
    # Save coin_collection_training_data
    with open(path + "coin_collection_training_data_" + str(len(coin_collection_training_data)) + "_" + timestamp + ".pkl", 'wb') as file:
        pickle.dump(coin_collection_training_data, file)

    # Save death_events_training_data
    with open(path + "death_events_training_data_" + str(len(death_events_training_data)) + "_" + timestamp + ".pkl", 'wb') as file:
        pickle.dump(death_events_training_data, file)

    # Save kill_events_training_data
    with open(path + "kill_events_training_data_" + str(len(kill_events_training_data)) + "_" + timestamp + ".pkl", 'wb') as file:
        pickle.dump(kill_events_training_data, file)

    print(f"{len(coin_collection_training_data) + len(death_events_training_data) + len(kill_events_training_data)} training data were saved...")



    exit()



load_training_data = True
if load_training_data:
    import glob
    import pickle
    import numpy as np
    import random
    from agent_code.my_nn_agent.model.pytorch_model_alphazero import AlphaZeroNet
    from datetime import datetime
    import os
    coin_collection_training_data_file = glob.glob("data/training_data/*coin*.pkl")[0]
    death_events_training_data_file = glob.glob("data/training_data/*death*.pkl")[0]
    kill_events_training_data_file = glob.glob("data/training_data/*kill*.pkl")[0]
    with open(coin_collection_training_data_file, "rb") as file:
        coin_collection_training_data = pickle.load(file)
    with open(death_events_training_data_file, "rb") as file:
        death_events_training_data = pickle.load(file)
    with open(kill_events_training_data_file, "rb") as file:
        kill_events_training_data = pickle.load(file)

limit = min(len(coin_collection_training_data), len(death_events_training_data), len(kill_events_training_data))

training_data = random.sample(coin_collection_training_data, limit) + random.sample(death_events_training_data, limit) + random.sample(kill_events_training_data, limit)
len(training_data)


normalize_policy_probs = True
if normalize_policy_probs:
    wrong_policy_counter = 0
    c = (len(training_data[0][0]) - 1) // 2
    for i, (state, action_probs, reward) in enumerate(training_data):
        mask = np.zeros(6)
        is_occupied_big = np.any(state[[0, 1, 4, 6]], axis=0).astype(int)
        is_occupied = is_occupied_big[c - 1:c + 2, c - 1:c + 2]

        if is_occupied[1, 0] == 0:
            mask[0] = 1
        if is_occupied[0, 1] == 0:
            mask[1] = 1
        if is_occupied[1, 2] == 0:
            mask[2] = 1
        if is_occupied[2, 1] == 0:
            mask[3] = 1
        if state[2, 0, 0] == 1:
            mask[4] = 1
        mask[5] = 1
        # print(action_probs)
        if mask[np.argmax(action_probs)] == 0:
            wrong_policy_counter += 1

        action_probs *= 1.3
        action_probs = np.exp(action_probs - np.max(action_probs))
        action_probs[4] *= 2
        action_probs[5] /= 2
        action_probs *= mask
        action_probs /= action_probs.sum()

        training_data[i] = (state, action_probs, reward)
    print(f"Number of faulty action probs (happens if the action taken is not valid): {str(wrong_policy_counter)}")

consider_rotated_data = True
if consider_rotated_data:
    # Action encoder and rotation logic
    action_to_index = {"UP": 0, "LEFT": 1, "DOWN": 2, "RIGHT": 3, "BOMB": 4, "WAIT": 5}
    rotate_action_map = {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5}  # Rotates actions 90 degrees clockwise
    index_array = np.array([1,2,3,0,4,5])


    # Initialize lists to store rotated state collections
    training_data_rotated_90 = []
    training_data_rotated_180 = []
    training_data_rotated_270 = []

    # Generate rotated data
    for encoded_state, action_probs, reward in training_data:
        # Rotate the 3D encoded state matrix
        encoded_state_90 = np.rot90(encoded_state, k=1, axes=(1, 2))
        encoded_state_180 = np.rot90(encoded_state_90, k=1, axes=(1, 2))
        encoded_state_270 = np.rot90(encoded_state_180, k=1, axes=(1, 2))

        # Rotate action probabilities accordingly
        action_probs_90 = action_probs[index_array]
        action_probs_180 = action_probs_90[index_array]
        action_probs_270 = action_probs_180[index_array]

        # Append rotated states and corresponding action probabilities and rewards to lists
        training_data_rotated_90.append((encoded_state_90, action_probs_90, reward))
        training_data_rotated_180.append((encoded_state_180, action_probs_180, reward))
        training_data_rotated_270.append((encoded_state_270, action_probs_270, reward))

    # Combine all rotated states back into the main collection
    training_data += training_data_rotated_90 + training_data_rotated_180 + training_data_rotated_270

    # Output the length of encoded state collections to verify augmentation
    print(f"Original encoded states count: {len(training_data) // 4}")
    print(f"Rotated 90 degrees states count: {len(training_data_rotated_90)}")
    print(f"Rotated 180 degrees states count: {len(training_data_rotated_180)}")
    print(f"Rotated 270 degrees states count: {len(training_data_rotated_270)}")
    print(f"Total encoded states count after augmentation: {len(training_data)}")



check_if_there_is_faulty_move = False
if check_if_there_is_faulty_move:
    wrong_policy_counter = 0
    c = (len(training_data[0][0]) - 1) // 2
    for state, action_probs, reward in training_data:
        mask = np.zeros(6)
        is_occupied_big = np.any(state[[0, 1, 4, 6]], axis=0).astype(int)
        is_occupied = is_occupied_big[c - 1:c + 2, c - 1:c + 2]

        if is_occupied[1, 0] == 0:
            mask[0] = 1
        if is_occupied[0, 1] == 0:
            mask[1] = 1
        if is_occupied[1, 2] == 0:
            mask[2] = 1
        if is_occupied[2, 1] == 0:
            mask[3] = 1
        if state[2, 0, 0] == 1:
            mask[4] = 1
        mask[5] = 1
        # print(action_probs)
        if mask[np.argmax(action_probs)] == 0:
            wrong_policy_counter += 1

    print(f"Number of faulty action probs in Total amongs {str(len(training_data))} training data (happens if the action taken is not valid): {str(wrong_policy_counter)}")


import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_or_validate(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        memory: list,
        args: dict,
        is_train: bool = True,
        verbose: bool = False
) -> tuple:
    """
    Performs either training or validation on the given model using the provided memory.

    :param model: The neural network model to be trained or validated.
    :param optimizer: The optimizer for training the model. It should be None if is_train is False.
    :param memory: A list of tuples containing the states, policy targets, and value targets for training or validation.
    :param args: A dictionary of arguments including batch size and other hyperparameters.
    :param is_train: A flag indicating whether the function should perform training (True) or validation (False).
    :param verbose: A flag to indicate whether to print detailed loss information per batch.
    :return: A tuple containing the average total loss, average policy loss, and average value loss per batch.
    """
    if not memory:
        raise ValueError("Memory is empty. Cannot perform training or validation.")

    random.shuffle(memory)
    total_loss, policy_total_loss, value_total_loss = 0.0, 0.0, 0.0
    num_batches = 0

    batch_size = args.get('batch_size', 256)

    for batch_start in range(0, len(memory), batch_size):
        batch_data = memory[batch_start:batch_start + batch_size]
        states, policy_targets, value_targets = zip(*batch_data)

        # Convert to numpy arrays and then to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=model.device)
        policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=model.device)
        value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=model.device)

        # Forward pass
        out_policy, out_value = model(states)

        # Compute losses
        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        policy_total_loss += policy_loss.item()
        value_total_loss += value_loss.item()
        num_batches += 1

        if verbose:
            print(
                f"Batch {num_batches} - Policy Loss: {policy_loss.item():.4f}, "
                f"Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}"
            )

    avg_total_loss = total_loss / num_batches
    avg_policy_loss = policy_total_loss / num_batches
    avg_value_loss = value_total_loss / num_batches

    return avg_total_loss, avg_policy_loss, avg_value_loss


# Model and optimizer initialization
model = AlphaZeroNet("AlphaZeroNet_V0", device="mps", planes=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Training arguments and memory split
args = {
    'patience': 2,
    'num_epochs': 10,
    'batch_size': 256,
}


random.shuffle(training_data)  # shuffle memory
# Calculate split index
split_idx = int(len(training_data) * 0.8)
# Split memory
train_memory = training_data[:split_idx]
val_memory = training_data[split_idx:]
# from earlystopping import EarlyStopping

# early_stopping_criterion = EarlyStopping(patience=args['patience'], verbose=True, delta=0.003,
#                                          save_multiple_models=True, min_lr=0.0001, model_iter=4)

# Training and validation loop
for epoch in tqdm(range(args['num_epochs']), desc="Training Progress"):
    model.train()
    train_loss, train_policy_loss, train_value_loss = train_or_validate(
        model, optimizer, train_memory, args, is_train=True, verbose=False
    )
    print(
        f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
        f"Policy Loss: {train_policy_loss:.4f}, Value Loss: {train_value_loss:.4f}"
    )

    model.eval()
    with torch.no_grad():  # Disable gradient calculation for validation
        val_loss, val_policy_loss, val_value_loss = train_or_validate(
            model, optimizer, val_memory, args, is_train=False, verbose=False
        )
    print(
        f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, "
        f"Policy Loss: {val_policy_loss:.4f}, Value Loss: {val_value_loss:.4f}"
    )
    # early_stopping_criterion(val_loss, model, optimizer)  # check if early stopping condition is met

    # if early_stopping_criterion.early_stop:
    #     break

# torch.save(early_stopping_criterion.model_state_dict, "data/model/model_sm4.pt")
# torch.save(early_stopping_criterion.optimizer_state_dict, "data/model/optimizer_sm4.pt")
torch.save(model.state_dict(), "agent_code.my_nn_agent.model.model2.pt")

import pickle
pickle_file_path = 'data/my_dict.pkl'
with open(pickle_file_path, "wb") as f:
    pickle.dump(training_data[0], f)
    print(f"Dictionary saved to {pickle_file_path}")



import pickle
import torch
from agent_code.my_nn_agent.model.pytorch_model_alphazero import AlphaZeroNet
import numpy as np
pickle_file_path = 'data/my_dict.pkl'
with open(pickle_file_path, 'rb') as f:
    loaded_dict = pickle.load(f)
    print("Loaded dictionary:", loaded_dict)
states, policy_targets, value_targets = loaded_dict
m = AlphaZeroNet()

# checkpoint = torch.load(model_optimizer_path)
m.load_state_dict(torch.load("data/model/model_sm1.pt"))
m.eval()
with torch.no_grad():  # Disable gradient calculation for validation
    states = torch.tensor(np.array(states), dtype=torch.float32, device=m.device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=m.device)
    value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=m.device)
    out_policy, out_value = m(states.unsqueeze(0))
action_prob = out_policy.squeeze().numpy()

action_prob_move_hist = [(action_prob, 2), (action_prob+1,3), (action_prob+4, 5)]

values_to_mask = [y for (x,y) in action_prob_move_hist if np.all(x == action_prob+11)]
if values_to_mask:
    action_probs[values_to_mask] = 0
if len(action_prob_move_hist) > 9:
    action_prob_move_hist.pop(0)
