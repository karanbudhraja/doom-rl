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

import agents

#
# initialization
#

# learing settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

batch_size = 64
test_episodes_per_epoch = 100

# other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False

# configuration file path
configuration_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

def preprocess_image_data(image_data):
    # downsample image
    # add dimension to align with having multiple channels
    image_data = skimage.transform.resize(image_data, resolution)
    image_data = image_data.astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)

    return image_data

def create_game():
    game = vzd.DoomGame()
    game.load_config(configuration_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    return game

def test(game, agent):
    # run test episodes and print result
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess_image_data(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Test results: mean: %.1f +/- %.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    # run training episodes
    # skip a few frames after each action
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess_image_data(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess_image_data(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        # training results
        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        # testing results
        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)

    game.close()
    return agent, game

if __name__ == '__main__':
    # initialize game and get all possible actions
    game = create_game()
    number_of_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=number_of_buttons)]

    # Initialize our agent with the set parameters
    # use gpu if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    agent = agents.DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model, device=device)

    # Run the training for the set number of epochs
    agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                        steps_per_epoch=learning_steps_per_epoch)

    # Reinitialize the game with window visible
    game.close()
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
