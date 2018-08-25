"""
 multi_armed_bandit_EC.py  (author: Emiliano Cancellieri / git: ecancellieri)
 Solution of the multi-armed bandit problem following R.S.Sutton and A.G.Barto
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1236)  # for reproducibility

# =========================
# Define functions
# =========================

def build_initial_state(bandits):
    # number of time an bandit is chosen
    k = np.zeros(len(bandits), dtype=np.int)
    # Q function
#    Q = np.zeros(len(bandits), dtype=np.float)
    Q = np.random.random(len(bandits))
    return Q, k

def get_reward(bandits,action):
    reward = bandits[action][0]+(np.random.random()-0.5)*bandits[action][1]
    return reward

def refresh_Q_k(Q,k,action,reward):
    k[action] = k[action] + 1
    Q[action] = Q[action] + (1/k[action]) * (reward - Q[action])
    return Q, k

def take_action(bandits,epsilon,Q):
    rand = np.random.random()
    if rand < epsilon:
        # randomly explore a bandit
        action_explore = np.random.randint(len(bandits))
        return action_explore
    else:
        # greedly choose the best bandit
        action_greedy = np.argmax(Q)
        return action_greedy

def make_the_learning(bandits,epochs,
                      epsilon,Q,k,):
    action_history = []
    reward_history = []
    for i in range(epochs):
        # chose and action based on functin Q
        action = take_action(bandits,epsilon,Q)
        # take the reward of the action
        reward = get_reward(bandits,action)
        # update the Q function based on the reward
        Q, k = refresh_Q_k(Q,k,action,reward)
        action_history.append(action)
        reward_history.append(reward)
    return np.array(action_history),np.array(reward_history)

def action_plot(actions,epochs,experiments,bandits):
    frequency_of_actions = np.zeros((epochs,len(bandits)))
    for i in range(epochs):
        unique, count = np.unique(actions[:,i], return_counts=True)
        frequency_of_actions[i,unique] = count/experiments

    for i in range(len(bandits)):
        plt.plot(frequency_of_actions[:,i],label='Bandit %d'%i)
    plt.ylim(-0.1,1.1)
    plt.xlabel('Number of epochs')
    plt.ylabel('% of times picked')
    plt.legend()
    plt.show()

# =========================
# Settings
# =========================
bandits = [[0.30, 0.20],  # mean reward
           [0.60, 1.99],  # and reward sigma
           [0.15, 0.10],
           [0.10, 0.10], 
           [0.55, 0.50],
           [0.20, 0.40]]  

epochs = 10000     # number of tries done by the agend
                  # on each experiment
experiments = 50  # number of experiments
epsilon = 0.1     # probability of random exploration

# =========================
# Main for multi-armed bandit
# with variable gain
# =========================

actions = []
rewards = []
for i in range(experiments):
    Q, k = build_initial_state(bandits)
    action_history, reward_history = make_the_learning(bandits,epochs,epsilon,Q,k,)
    actions.append(action_history)
    rewards.append(reward_history)
actions = np.asarray(actions)
rewards = np.asarray(rewards)

action_plot(actions,epochs,experiments,bandits)




