from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import environment
import gc
import agent
import train
import buffer
import time
import pandas as pd
import matplotlib.pyplot as plt

env = environment.Environment()
# env = gym.make('Pendulum-v0')

MAX_EPISODES = 5000
MAX_STEPS = 100
MAX_BUFFER = 100000
#MAX_TOTAL_REWARD = 300
S_DIM = 3
A_DIM = 1
A_MAX = 200

print (' State Dimensions :- ' + str(S_DIM))
print (' Action Dimensions :- ' + str(A_DIM))
print (' Action Max :- ' + str(A_MAX))

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

# temporary fixed utility
u1 = 100
pos_reward = 0
neg_reward = 0
rewards = []

for _ep in range(MAX_EPISODES):
	print ('EPISODE :- ' + str(_ep))
	random_agent = agent.Agent()
	u2 = random_agent.utility
	print("Random Player utility: %f"%u2)
	observation, cur_bid = env.reset(u1, u2)
	bid1 = 0
	bid2 = 0
	total_reward = 0
	for r in range(MAX_STEPS):
		state = np.float32(observation)
		# random action interaction
		print("=================Random Agent Turn=================")
		action = random_agent.action(cur_bid)
		print("Action taken: %f"%action)
		bid2 = action
		new_observation, reward, done, cur_bid = env.step(bid1, bid2,idx=2)
		total_reward += reward
		print("===============Feedback to learned agent round===============")
		print("Observation:")
		print(new_observation)
		print("Reward: %f, Currnt Bid: %f"%(reward, cur_bid))

		new_state = np.float32(new_observation)
		# push this exp in ram
		ram.add(state, action, reward, new_state)

		observation = new_observation

		trainer.optimize()

		print("Is done? "+str(done))

		# perform optimization
		if done:
			break

		# AC agent interaction
		print("=================Learned Agent Turn=================")
		#TODO: check
		if _ep%5 == 0:
			# validate every 5th episode
			action = trainer.get_exploitation_action(state)
			print("Exploit action: %f"%action)
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(state)
			print("Explore action: %f"%action)

		print("Action taken: %f"%action)
		bid1 = action
		_, _, done, cur_bid = env.step(bid1, bid2, idx=1)
		print("===============Feedback to random agent round===============")
		print("Currnt Bid: %f"%cur_bid)
		# print out information
		# action = trainer.get_exploration_action(state)

		# opponent action
		# TODO: random agent
		
		# attr: self info


		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue

	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)
	rewards.append(total_reward)

	print("Episode End")
	if reward > 0:
		pos_reward += 1
	elif env.u1 > env.u2 and reward <= 0:
		neg_reward += 1

	print("Positive: %d, Negative: %d"%(pos_reward, neg_reward))

	if _ep%100 == 0:
		trainer.save_models(_ep)

#print(rewards)
print ('Completed episodes')
print("Positive Reward Porpotion: %f"%(pos_reward / (pos_reward + neg_reward)))
plt.plot(np.array(rewards), '--')
plt.show()