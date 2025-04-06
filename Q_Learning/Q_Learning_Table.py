# Importing dependencies
import gymnasium
import numpy as np

env = gymnasium.make("MountainCar-v0")   # Creating an instance of the environment, on which our model will be training.
env.reset() # Initializing the environment

# Getting some information from this environment
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)  # It will tell us all the actions that our model can take inside this env.

# Creating a discrete observation size
discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size
print(discrete_os_win_size)

epsilon = 0.5 # It is the measure of how much randomness we want when getting the q-values. It also indicates for how long we want our model to be in exploration phase.
# We will be decreasing the value of epsilon as we go further into training the model. Epsilon is important as it will allow the model to test out different pathsto achieve the final objective and enable it to choose the most optimal path.

q_table = np.random.uniform(low = -2, high = 0, size = (discrete_os_size + [env.action_space.n]))

# Creating a function which will convert the new_state we get when we do env.step(), which are continuous and we need to convert them into discrete values
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))


learning_rate = 0.01
discount_factor = 0.93   # It is a measure of how important we find future actions. It is typically between 0.9 and 0.99 the higher its value the more importance the agent will give to the future actions instead of just completing the current stage/level.
episodes = 25000

start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)  # The amount we want to decay our epsilon each episode.

check = 2000

for episode in range(episodes):
    if episode % check :
        print(episode)
        render = True
    else :
        render = False

    if render:
        env.render()

    done = False

    discrete_state = get_discrete_state(env.reset())

    while not done:
        # Using the epsilon to introduce randomness to our agent actions.
        if np.random.random() > epsilon:   # np.random.random generates a random integer between 0 and 1.
            action = np.argmax(q_table[discrete_state])  # Taking the action with the highest q-value using the argmax function.
        else :
            action = np.random.randint(0, env.action_space.n)  # Creating a random q-table containing random actions.
        new_state, reward, done, _ = env.step(action = action)

        new_discrete_state = get_discrete_state(new_state)
        env.render()  # So, that we can see the game
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)  # The backpropagation formula for q-learning.

            # Updating the q-table with the new q-value
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] > env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon += epsilon_decay_value
env.close()

# The q table is a table contain various q-values which are combination of various action being taken in the env on the basis of these q-values our model will be making decisions.
# Initially these q values will be chosen at random and thus this phase will contain just random actions. This phase is thus, called the exploration phase.

# Mehtods to analyse the agent:
# The most common way, which works in most scenarios is by tracking the rewards.