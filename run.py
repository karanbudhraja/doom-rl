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
    random_agent = RandomAgent(actions)

    # iteration
    episodes = 1
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE
    for index in range(episodes):
        # start new episode
        game.new_episode()

        while not game.is_episode_finished():
            # get current state
            state = game.get_state()

            # Which consists of:
            state_number = state.number
            game_variables = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            # take an action
            current_action = random_agent.get_action()

            # get reward
            reward = game.make_action(current_action)
            current_total_reward = game.get_total_reward()

            # logging
            print("Episode #" + str(index + 1))
            print("State #" + str(state_number))
            print("Game variables:", game_variables)
            print("Reward:", reward)
            print("Total reward:", current_total_reward)
            print("=====================")

            # sleep
            sleep(sleep_time)

        # episode results
        print("Episode total reward:", current_total_reward)

    # cleanup
    game.close()

if __name__ == "__main__":
    main()
