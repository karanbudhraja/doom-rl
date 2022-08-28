#
# custom agents added here
#

import itertools
import numpy as np
from time import sleep
import vizdoom as vzd
import agents

import os
import pickle
import matplotlib.pyplot as plt

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
    random_agent = agents.RandomAgent(actions)
    policy_agent = agents.PolicyAgent(actions, input_size, data_directory_name)

    # create instance and initialize
    game = get_game()

    #
    # iteration
    #

    iterations = 100
    episodes_per_iteration = 5
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE
    iteration_average_loss_values = []
    iteration_average_total_reward_values = []

    for iteration_index in range(iterations):

        #
        # interact with the world and gather data
        #

        print("Iteration", iteration_index+1)

        # gather data
        for episode_index in range(episodes_per_iteration):
            # start new episode
            episode_file_name = str(episode_index).zfill(4) + data_file_extension
            episode_file_path = os.path.join(data_directory_name, episode_file_name)
            episode_data = []
            total_rewards = []

            game.init()
            game.new_episode()
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

                # record data
                current_data = {"state": state.screen_buffer, "action": action, "reward": reward, "next_state": next_state.screen_buffer}
                episode_data.append(current_data)

                # sleep
                # sleep(sleep_time)

            # save episode data
            with open(episode_file_path, "wb") as episode_data_file:
                pickle.dump(episode_data, episode_data_file)

            # episode results
            total_rewards.append(total_reward)

        #
        # update model
        #

        iteration_average_loss = policy_agent.update()
        iteration_average_total_reward = np.mean(total_rewards)
        iteration_average_loss_values.append(iteration_average_loss)
        iteration_average_total_reward_values.append(iteration_average_total_reward)

        # clean directory
        for episode_file_name in os.listdir(data_directory_name):
            episode_file_path = os.path.join(data_directory_name, episode_file_name)
            os.remove(episode_file_path)

    # cleanup
    game.close()

    # display information
    plt.figure()
    plt.plot(iteration_average_loss_values)
    plt.savefig(os.path.join(log_directory_name, "iteration_average_loss_values.pdf"))
    plt.figure()
    plt.plot(iteration_average_total_reward_values)
    plt.savefig(os.path.join(log_directory_name, "iteration_average_total_reward_values.pdf"))

if __name__ == "__main__":
    main()
