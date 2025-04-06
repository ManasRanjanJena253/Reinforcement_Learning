# For training a DQN we need to show it a similar or different experiences again and again to the DQN so, that it can learn efficiently.
# The experience replay typically consists of :
# state, action, new_state, reward, terminated
# For this we will create a python deque, which is simply a python list from which we can take out or input data from both the end and start of the list. It is a first in first out situation.

# Importing dependencies
from collections import deque
import random

# Creating ReplayMemory class
class ReplayMemory():
    def __init__(self, maxlen, seed = None):
        self.memory = deque([], maxlen = maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)