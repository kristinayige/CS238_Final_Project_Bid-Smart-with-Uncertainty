from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import environment
import gc
import agent
import train
import buffer
import pandas as pd
import matplotlib.pyplot as plt

env = environment.Environment(t=1)

u1 = 100

MAX_EPISODES = 2500
MAX_STEPS = 100
MAX_BUFFER = 100000
S_DIM = 2
A_DIM = 1
A_MAX = 200
ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

u1 = 100

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
		state = np.float32(observation)
		# random action interaction
		# print("=================Random Agent Turn=================")
		action = random_agent.action(cur_bid)
		# print("Action taken: %f"%action)
		bid2 = action
		new_observation, reward, done, cur_bid = env.step(bid1, bid2,idx=2)
		total_reward += reward
		# print("===============Feedback to learned agent round===============")
		# print("Observation:")
		# print(new_observation)
		# print("Reward: %f, Currnt Bid: %f"%(reward, cur_bid))

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
		# print("=================Learned Agent Turn=================")
		#TODO: check
		if _ep%5 == 0:
			# validate every 5th episode
			action = trainer.get_exploration_action(state)
			# print("Exploit action: %f"%action)
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploitation_action(state)
			# print("Explore action: %f"%action)

		# print("Action taken: %f"%action)
		bid1 = action
		_, _, done, cur_bid = env.step(bid1, bid2, idx=1)
		# print("===============Feedback to random agent round===============")
		# print("Currnt Bid: %f"%cur_bid)

	# check memory consumption and clear memory
	gc.collect()