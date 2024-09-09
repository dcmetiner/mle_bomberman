from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
COIN_COLLECTED_EVENT = "COIN_COLLECTED"
INVALID_ACTION = "INVALID_ACTION"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.total_rewards = 0
    self.total_rewards_list = []

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
    events_log = ", ".join(map(repr, events)) 

    # self.logger.debug(f'Encountered game event(s) {events_log} in step {new_game_state["step"]}')
    
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    # epsilon = 0.5
    # epsilon_decay = 0.995
    # min_epsilon = 0.01

    # self.logger.debug(f'{old_game_state["explosion_map"]}')

    # Idea: Add your own events to hand out rewards
    # if(self_action.)
    # if(self_action == )
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    last_transition = get_last_transition(self)
    
    if last_transition:
        state = last_transition.state
        state_key = state
        
        next_state = last_transition.next_state
        # if isinstance(next_state, (list, tuple, np.ndarray)):
        next_state_key = next_state
        action = last_transition.action
        # self.logger.info(f"last transition action: {action}")
        if action is None:
            # self.logger.info(f"action is none {action}")
            action = 'WAIT'
        reward = last_transition.reward
        self.total_rewards += reward
        # self.logger.debug(f"reward = {reward}")
        if state_key not in self.model:
            self.model[state_key] = np.zeros(len(ACTIONS))
        if next_state_key in self.model:
            best_next_q_value = np.max(self.model[next_state_key])
            # self.logger.debug(f"best_next_q_value = {best_next_q_value}")
        else:
            best_next_q_value = 0
        self.model[state_key][ACTIONS.index(action)] += alpha * (reward + gamma * best_next_q_value - self.model[state_key][ACTIONS.index(action)])

    
def get_last_transition(self):
    """
    Retrieve the last transition from the transitions list.

    :return: The last Transition object or None if no transitions are available.
    """
    if self.transitions:
        return self.transitions[-1]
    else:
        return None


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
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Initialize the rewards list

    # Assume `self.total_rewards` is being updated correctly in your episode loop
    # After completing an episode, update the rewards list
    self.total_rewards_list.append(self.total_rewards)
    self.total_rewards = 0  # Reset for the next episode

    # Plotting
    plt.figure(figsize=(10, 6))

    # Check if total_rewards_list is not empty to avoid crashes
    if self.total_rewards_list:
        plt.plot(self.total_rewards_list, label="Total Reward")
        plt.title("Total Rewards Per Episode (Bomberman)")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.legend()

        # Save the plot as an image file
        plt.savefig("total_rewards_per_episode.png", format='png')

        plt.close()
    else:
        print("No rewards to plot.")

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -30,
        e.GOT_KILLED: -50,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        INVALID_ACTION: -.5,
        e.CRATE_DESTROYED: 3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
