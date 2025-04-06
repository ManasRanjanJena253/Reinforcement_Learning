# Importing dependencies
import os

import flappy_bird_gymnasium
import torch
import torch.nn as nn
import random
import gymnasium
from Implementing_DQN_with_pytorch import DQN
from Implementing_experience_Replay import ReplayMemory
import itertools    # This is used for running a loop indefinitely.
import yaml
import pygame
import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import numpy as np

runs_dir = "runs"
os.makedirs(runs_dir, exist_ok = True)

matplotlib.use('Agg')   # Agg is used to generate plots as images and save them to a file instead of rendering it to screen.
class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:   # Opening the yml file containing the hyperparameters stored in it.
            all_hyperparameter_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_set[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.network_sync_rate = hyperparameters['network_sync_rate']

        self.last_graph_update_time = datetime.datetime.now()

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        # Path to run info
        self.model_file = os.path.join(runs_dir, "flappy_bird.pt")
        self.graph_file = os.path.join(runs_dir, "graphs.png")

    def run(self, is_training = True, render = False):

        # Creating an instance of the flappy bird env.
        env = gymnasium.make("FlappyBird-v0", render_mode = "human" if render else None)
        # Setting the mode to render_mode = "human", to view the game on screen.
        # use_lidar is a custom parameter which mean you need to turn lidar on or not.
        num_states = env.observation_space.shape[0]
        print(env.action_space)
        num_actions = env.action_space.n


        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(state_dim = num_states, action_dim = num_actions)
        memory = ReplayMemory(10000)
        epsilon = self.epsilon_init
        if is_training:
            target_dqn = DQN(state_dim=num_states, action_dim=num_actions)
            target_dqn.load_state_dict(policy_dqn.state_dict())# Copying the policy dqn as the actions will be taken on the basis of this dqn and if both
            # The policy dqn and target dqn will be same the constant change in policy dqn will make it unable to take actions in the new episode.

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate)

            step_counter = 0

            best_reward = -999999

        else:
            policy_dqn = torch.load(self.model_file, weights_only = False)
            policy_dqn.eval()

        for episode in range(10000):
            state, _ = env.reset()  # Calling the reset function to initialize the environment.
            state = torch.tensor(state, dtype = torch.float)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # Calling the sample function on the environment to get a random action.
                    step_counter += 1
                else:
                    with torch.inference_mode():
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax()
                # Next action:
                # (feeding the observation to the agent)
                # The actions for this environment i.e. the action space of this env include :
                # 0 - do nothing
                # 1 - flap

                # Processing :
                new_state, reward, terminated, _, info = env.step(action = int(action))

                # terminated checks whether the bird is still alive or not.
                # If not alive then True, otherwise False.

                # The rewards for each action :
                # 1. +0.1 - every frame it stays alive
                # 2. +1.0 - successfully passing the pipe
                # 3. -1.0 - dying
                # 4. -0.5 - touch the top of the screen

                # Changing the reward structure of the flappy bird
                if reward == 0.1:
                    reward = 2
                if reward == 1:
                    reward = 20
                if reward == -1:
                    reward = -20
                if reward == -0.5:
                    reward = -15
                new_state = torch.tensor(new_state, dtype = torch.float)
                reward = torch.tensor(reward, dtype = torch.float)

                # Accumulating the rewards
                episode_reward += reward
                memory.append((state, torch.tensor(action, dtype = torch.int64), new_state, reward, terminated))
                state = new_state
            reward_per_episode.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
            if is_training:
                # Updating the graph every x seconds
                curr_time = datetime.datetime.now()
                if curr_time - self.last_graph_update_time > timedelta(seconds = 1):
                    self.save_graph(reward_per_episode, epsilon_history)
                    self.last_graph_update_time = curr_time
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)
                step_counter += 1
                if step_counter > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_counter = 0

        env.close()
        torch.save(policy_dqn, self.model_file)

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99): x + 1])

        # Plot mean rewards
        axs[0].set_title("Mean Rewards")
        axs[0].set_ylabel("Rewards")
        axs[0].plot(mean_rewards)

        # Plot epsilon decay
        axs[1].set_title("Epsilon Decay")
        axs[1].set_ylabel("Epsilon")
        axs[1].plot(epsilon_history)

        plt.tight_layout()
        fig.savefig(self.graph_file)
        plt.close(fig)
        print('graph saved')


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, new_state, reward, terminated in mini_batch:
            current_q = policy_dqn(state)
            if terminated:
                target_q = reward.unsqueeze(0)
            else:
                target_q = reward + self.discount_factor * target_dqn(new_state).max().detach()
                current_q = policy_dqn(state)


            # Computing the loss for the whole mini batch
            loss = self.loss_fn(current_q, target_q)

            # Optimizing the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
if __name__ == '__main__':
    agent = Agent('cartpole1')
    agent.run(is_training = True, render = False)



