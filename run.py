#
# custom agents added here
#

import itertools
import numpy as np
from time import sleep
import vizdoom as vzd
import agents

import os
import matplotlib.pyplot as plt
import skimage.transform
import shutil

def get_game():
    # create and configure a game instance
    game = vzd.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.set_screen_format(vzd.ScreenFormat.GRAY8)

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

def get_state_data(state):
    # extract state data from state object
    state_data = state.screen_buffer.astype(np.float32)
    state_data = skimage.transform.resize(state_data, (30, 45))
    state_data = np.expand_dims(state_data, axis=0)

    return state_data

def main():
    #
    # initialization
    #

    # directory dependencies
    log_directory_name = "logs"
    data_directory_name = "data_buffer"
    os.makedirs(log_directory_name, exist_ok=True)
    os.makedirs(data_directory_name, exist_ok=True)
    data_file_extension = ".npy"
    states_file_name = "states" + data_file_extension
    action_policies_file_name = "action_policies" + data_file_extension
    rewards_file_name = "rewards" + data_file_extension
    next_states_file_name = "next_states" + data_file_extension

    # create instance and initialize
    # collect action choices
    game = get_game()
    game.init()
    actions = get_all_possible_action_combinations(game)
    input_size = (game.get_screen_channels(), game.get_screen_height(), game.get_screen_width())
    game.close()

    # define agents
    # agent = agents.RandomAgent(len(actions))
    agent = agents.PolicyAgent(input_size, len(actions), data_directory_name,
                                states_file_name, action_policies_file_name, rewards_file_name, next_states_file_name)

    # create instance and initialize
    game = get_game()

    #
    # iteration
    #

    iterations = 2
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
            episode_directory_name = str(episode_index).zfill(4)
            episode_directory_path = os.path.join(data_directory_name, episode_directory_name)
            os.makedirs(episode_directory_path, exist_ok=True)
            total_rewards = []

            game.init()
            game.new_episode()
            states = []
            action_policies = []
            rewards = []
            next_states = []
            while not game.is_episode_finished():
                # get current state
                state = game.get_state()

                # take an action
                policy = agent.get_policy(np.expand_dims(get_state_data(state), axis=0), episode_index+1).clone().detach().numpy()
                action = actions[np.argmax(policy)]
                
                # get next state and action reward
                next_state = game.get_state()
                reward = game.make_action(action)

                # record data
                states.append(get_state_data(state))
                action_policies.append(policy)
                rewards.append(reward)
                next_states.append(get_state_data(next_state))

            # save episode data
            states = np.stack(states)
            action_policies = np.stack(action_policies)
            rewards = np.stack(rewards)
            next_states = np.stack(next_states)
            np.save(os.path.join(episode_directory_path, "states"), states)
            np.save(os.path.join(episode_directory_path, "action_policies"), action_policies)
            np.save(os.path.join(episode_directory_path, "rewards"), rewards)
            np.save(os.path.join(episode_directory_path, "next_states"), next_states)

            # episode results
            total_reward = game.get_total_reward()
            total_rewards.append(total_reward)

        #
        # update model
        #

        iteration_average_loss = agent.update()
        iteration_average_total_reward = np.mean(total_rewards)
        iteration_average_loss_values.append(iteration_average_loss)
        iteration_average_total_reward_values.append(iteration_average_total_reward)

        # clean directory
        for episode_file_name in os.listdir(data_directory_name):
            episode_file_path = os.path.join(data_directory_name, episode_file_name)
            shutil.rmtree(episode_file_path)

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
