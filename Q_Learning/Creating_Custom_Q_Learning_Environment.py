# Importing dependencies
import gymnasium
from PIL import Image
import cv2
from matplotlib import style
import pickle
import time
import numpy as np
from tqdm.auto import tqdm
style.use("ggplot")

size = 10  # The no. of actions our action space will have
episodes = 10000
move_penalty = 1
enemy_penalty = 300
food_reward = 25

epsilon = 0.9
eps_decay = 0.6

show_every = 500
start_q_table = None

learning_rate = 0.1
discount_factor = 0.95

player_no = 1
food_no = 2
enemy_no = 3

# Setting up the colors in BGR Format
d = {1 : (255, 175, 0),  # Color of player
     2 : (0, 255, 0),    # Color of food
     3 : (0, 0, 255)}    # Color of enemy


class Blob:
    def __init__(self):
        # Setting the coordinates for movement of the player and the enemy.
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = 1, y = -1)

    def move(self, x = False, y = False):
        if not x :
            self.x += np.random.randint(-1, 2)   # Random movement if x is not given
        else:
            self.x += x

        if not y :
            self.y = np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > size-1:
            self.x = size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > size-1:
            self.y = size - 1

if start_q_table is None:
    q_table = {}
    for x1 in range(-size + 1, size):
        for y1 in range(-size + 1, size):
            for x2 in range(-size + 1, size):
                for y2 in range(-size + 1, size):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []
for episode in tqdm(range(episodes)):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % show_every == 0:
        print(f"on : {episode}, epsilon : {epsilon}")
        show = True
    else:
        show = False

    episode_rewards = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -enemy_penalty

        elif player.x == food.x and player.y == food.y:
            reward = food_reward

        else:
            reward = -move_penalty

        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == food_reward:
            new_q = food_reward
        elif reward == enemy_penalty:
            new_q = -enemy_penalty
        else:
            new_q = (1 - learning_rate) * (current_q + learning_rate * (reward + discount_factor * max_future_q))

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((size, size, 3), dtype = np.uint8)
            env[food.x][food.y] = d[food_no]
            env[player.x][player.y] = d[player_no]
            env[enemy.x][enemy.y] = d[enemy_no]

            img = Image.fromarray(env, 'RGB')
            img = img.resize((500, 500))
            cv2.imshow("Game", np.array(img))
            if reward == food_reward or reward == enemy_penalty:
                if cv2.waitkey(500) & 0xff == ord("q"):
                    break
            else:
                if cv2.waitkey(1) & 0xff == ord("q"):
                    break

    epsilon *= eps_decay

