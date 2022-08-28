#
# custom agents added here
#

from random import choice
import itertools
import numpy as np
from time import sleep
import vizdoom as vzd
from agents import *

import os
import pickle

def get_game():
    # create and configure a game instance
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")

    return game

def get_all_possible_action_combinations(game):
    # get all possible combinations of discrete actions
    action_space_size = game.get_available_buttons_size()
    actions = []
    for combination_size in range(0, action_space_size+1):
        for actions_subset in itertools.combinations(range(action_space_size), combination_size):
            current_action = np.array([False] * action_space_size)
            current_action[list(actions_subset)] = True
            actions.append(current_action)

    return actions

def main():
    #
    # initialization
    #

    # directory dependencies
    log_directory_name = "logs"
    data_directory_name = "data_buffer"
    os.makedirs(log_directory_name, exist_ok=True)
    os.makedirs(data_directory_name, exist_ok=True)
    data_file_extension = ".data"

    # create instance and initialize
    # collect action choices
    game = get_game()
    game.init()
    actions = get_all_possible_action_combinations(game)
    input_size = (game.get_screen_channels(), game.get_screen_height(), game.get_screen_width())
    game.close()

    # define agents
    random_agent = RandomAgent(actions)
    policy_agent = PolicyAgent(actions, input_size, data_directory_name)

    # create instance and initialize
    game = get_game()
    game.init()

    #
    # iteration
    #

    iterations = 5
    episodes_per_iteration = 3
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE
    
    for _ in range(iterations):

        #
        # interact with the world and gather data
        #

        for index in range(episodes_per_iteration):
            # start new episode
            episode_file_name = str(index).zfill(4) + data_file_extension
            episode_file_path = os.path.join(data_directory_name, episode_file_name)
            game.new_episode()
            episode_data = []

            while not game.is_episode_finished():
                # get current state
                state = game.get_state()

                # take an action
                # action = random_agent.get_action()
                action = policy_agent.get_action(state.screen_buffer)
                next_state = game.get_state()
                
                # get reward
                reward = game.make_action(action)
                total_reward = game.get_total_reward()

                current_data = {"state": state.screen_buffer, "action": action, "reward": reward}
                episode_data.append(current_data)

            # save episode data
            with open(episode_file_path, "wb") as episode_data_file:
                pickle.dump(episode_data, episode_data_file)

            # sleep
            # sleep(sleep_time)

            # episode results
            print("episode", index, "total reward:", total_reward)

        #
        # update model
        #

        policy_agent.update()

    # cleanup
    game.close()

if __name__ == "__main__":
    main()
