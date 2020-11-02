import agent
import environment
import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

env = environment.Environment()
learning_rate = 0.01
gamma = 0.99
MAX_EPISODES = 50000
MAX_STEPS = 10
ACTION_SIZE = 200 # per step 2 dollars
u1 = 100

# Q table contains states x acitons
# states contain 
Q = np.zeros((MAX_STEPS * ACTION_SIZE, ACTION_SIZE))

def state_index(time, cur_bid):
    return time * ACTION_SIZE + cur_bid

def epsilon_exploration(state_idx, cur_bid, epsilon):
    if random.uniform(0, 1) < epsilon:
        if random.uniform(0, 1) < 0.5:
            return cur_bid + 2
        else:
            return cur_bid + 4
    else:
        # obtain optimal action based on Q table
        act_idx = np.argmax(Q[state_idx, :])
        return act_idx * 2

def update_q_table(state_idx, action, reward, next_state_idx):
    next_val = np.max(Q[new_state_idx, :])
    
    action_idx = (int)(action / 2)
    curr_val = Q[state_idx, action_idx]

    Q[state_idx, action_idx] += learning_rate * (reward + gamma * next_val - curr_val)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

for _ep in range(MAX_EPISODES):
    print ('EPISODE :- ' + str(_ep))
    random_agent = agent.Agent()
    u2 = random_agent.utility
    observation, cur_bid = env.reset(u1, u2)
    bid1 = 0
    bid2 = 0
    action = 0
    total_reward = 0
    for _ in range(MAX_STEPS):
        time, old_bid = observation
        state_idx = state_index(time, old_bid)
        # print("=================Random Agent Turn=================")
        bid2 = random_agent.action(cur_bid)
        # print("Action taken: %f"%bid2)
        new_observation, reward, done, cur_bid = env.step(bid1, bid2,idx=2)
        total_reward += reward
        # print("===============Feedback to learned agent round===============")
        # print("Observation:")
        # print(new_observation)
        # print("Reward: %f, Currnt Bid: %f"%(reward, cur_bid))

        new_time, new_bid = new_observation
        new_state_idx = state_index(new_time, new_bid)

        # update q table
        update_q_table(state_idx, action, reward, new_state_idx)

        if done:
            break

        observation = new_observation

        # print("=================Learned Agent Turn=================")
        action = epsilon_exploration(new_state_idx, new_bid, epsilon)
        epsilon *= epsilon
        # print("Action: %f"%action)
        bid1 = action
        _, _, done, cur_bid = env.step(bid1, bid2, idx=1)
        # print("===============Feedback to random agent round===============")
        # print("Currnt Bid: %f"%cur_bid)


df = pd.DataFrame(Q)
pd.to_csv("q_table.csv")