#
# custom agents added here
#

from random import choice
import itertools
import numpy as np
from time import sleep
import vizdoom as vzd
from agents import *

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
    # create instance and initialize
    game = get_game()
    game.init()

    # collect action choices
    actions = get_all_possible_action_combinations(game)
    input_size = (game.get_screen_channels(), game.get_screen_height(), game.get_screen_width())

    # define agents
    random_agent = RandomAgent(actions)
    policy_agent = PolicyAgent(actions, input_size)

    # iteration
    episodes = 3
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE
    for index in range(episodes):
        # start new episode
        game.new_episode()

        while not game.is_episode_finished():
            # get current state
            state = game.get_state()

            #
            # state consists of:
            #
            # state.number
            # state.game_variables
            # state.screen_buffer
            # state.depth_buffer
            # state.labels_buffer
            # state.automap_buffer
            # state.labels
            # state.objects
            # state.sectors

            # take an action
            # action = random_agent.get_action()
            action = policy_agent.get_action(state)
            next_state = game.get_state()
            
            # get reward
            reward = game.make_action(action)
            total_reward = game.get_total_reward()

            # add to data buffer
            policy_agent.add_to_data_buffer(index, state, action, reward, total_reward, next_state) 

            # logging
            print("Episode #" + str(index + 1))
            print("State #" + str(state.number))
            print("Game variables:", state.game_variables)
            print("Reward:", reward)
            print("Total reward:", total_reward)
            print("=====================")

            # sleep
            # sleep(sleep_time)

        # episode results
        print("Episode total reward:", total_reward)

    # cleanup
    game.close()

if __name__ == "__main__":
    main()
