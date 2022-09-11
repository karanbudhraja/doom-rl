#
# library dependencies
#

import vizdoom as vzd
import torch
import numpy as np
import itertools as it
import skimage.transform

import os
from time import sleep
from tqdm import trange

import matplotlib.pyplot as plt

from agents import random_agents, q_learning_agents, policy_learning_agents

#
# functions
#

def preprocess_image_data(image_data, resolution=(30,45)):
    # downsample image
    # add dimension to align with having multiple channels
    image_data = skimage.transform.resize(image_data, resolution)
    image_data = image_data.astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)

    return image_data

def create_game(configuration_file_name="basic.cfg"):
    game = vzd.DoomGame()
    configuration_file_path = os.path.join(vzd.scenarios_path, configuration_file_name)
    game.load_config(configuration_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    return game

def test(game, agent, frame_repeat, test_episodes_per_epoch=100):
    # run test episodes and print result
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess_image_data(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Test results: mean: %.1f +/- %.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

def get_results(game, agent, frame_repeat, epoch_average_loss_values, epoch_average_train_scores, episodes_to_watch=10):
    #
    # plot data
    #

    plt.plot()
    plt.figure()
    plt.plot(epoch_average_loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.savefig(os.path.join(agent.log_directory_name, "iteration_average_loss_values.pdf"))
    plt.figure()
    plt.plot(epoch_average_train_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Average train score")
    plt.savefig(os.path.join(agent.log_directory_name, "iteration_average_total_reward_values.pdf"))

    #
    # demonstrate
    #

    # reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess_image_data(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

def run_random_sampling(game, actions, agent, frame_repeat=12, num_epochs=5, steps_per_epoch=2000, save_model=True):
    #
    # training
    #
    
    # run training episodes
    # skip a few frames after each action
    epoch_average_train_scores = []
    epoch_average_loss_values = []
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        loss_values = []
        for _ in trange(steps_per_epoch):
            state = preprocess_image_data(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess_image_data(game.get_state().screen_buffer)
            else:
                # padding in case the episode has been finished
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            # add to data buffer
            agent.append_memory(state, action, reward, next_state, done)

            if global_step >= agent.batch_size:
                loss = agent.train()
                loss_values.append(loss)

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        # update model
        # record data
        agent.update()
        train_scores = np.array(train_scores)

        # training results
        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        # testing results
        test(game, agent, frame_repeat)
        if save_model:
            print("Saving the network weights to:", agent.model_save_file_path)
            torch.save(agent.q_net, agent.model_save_file_path)

        # get epoch statistics
        epoch_average_loss_values.append(np.mean(loss_values))
        epoch_average_train_scores.append(np.mean(train_scores))

    game.close()

    # check performance and save results
    get_results(game, agent, frame_repeat, epoch_average_loss_values, epoch_average_train_scores)

def run_episodic_sampling(game, actions, agent, frame_repeat=12, num_epochs=2, episodes_per_epoch=100, episodes_to_watch=10, save_model=True):
    #
    # training
    #
    
    # run training episodes
    # skip a few frames after each action
    epoch_average_train_scores = []
    epoch_average_loss_values = []
    for epoch in range(num_epochs):
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        loss_values = []
        for episode in trange(episodes_per_epoch):
            game.new_episode()

            # remove previous episode data stored in data buffer
            # this is to avoid having unseen state transitions in our recording from overlapped episodes
            agent.clear_memory(episode)

            # play through episode
            while not game.is_episode_finished():
                state = preprocess_image_data(game.get_state().screen_buffer)
                action = agent.get_action(state)
                reward = game.make_action(actions[action], frame_repeat)
                done = game.is_episode_finished()

                if not done:
                    next_state = preprocess_image_data(game.get_state().screen_buffer)
                else:
                    # padding in case the episode has been finished
                    next_state = np.zeros((1, 30, 45)).astype(np.float32)

                # add to data buffer
                agent.append_memory(episode, state, action, reward, next_state, done)

            if (global_step >= agent.memory_size):
                loss = agent.train()
                loss_values.append(loss)

            train_scores.append(game.get_total_reward())
            global_step += 1

        # update model
        # record data
        agent.update()
        train_scores = np.array(train_scores)

        # training results
        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        # testing results
        test(game, agent, frame_repeat)
        if save_model:
            print("Saving the network weights to:", agent.model_save_file_path)
            agent.save()

        # get epoch statistics
        epoch_average_loss_values.append(np.mean(loss_values))
        epoch_average_train_scores.append(np.mean(train_scores))

    game.close()

    # check performance and save results
    get_results(game, agent, frame_repeat, epoch_average_loss_values, epoch_average_train_scores)

#
# main
#

if __name__ == '__main__':
    #
    # initialization
    #

    # initialize game and get all possible actions
    game = create_game()
    number_of_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=number_of_buttons)]

    # initialize agent
    # use gpu if available
    # run training and testing
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
 
    # agent = random_agents.RandomAgent(device, len(actions))
    # agent = q_learning_agents.QLearningAgent(device, len(actions), q_learning_agents.QNet, torch.nn.MSELoss())
    # agent = q_learning_agents.QLearningAgent(device, len(actions), q_learning_agents.DuelQNet, torch.nn.MSELoss())
    # run_random_sampling(game, actions, agent)

    # agent = policy_learning_agents.PolicyLearningAgent(device, len(actions), policy_learning_agents.PolicyNet, torch.nn.MSELoss())
    agent = random_agents.RandomAgent(device, len(actions))
    run_episodic_sampling(game, actions, agent)
