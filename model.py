import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,3)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())

		self.fca1 = nn.Linear(action_dim,1)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc1 = nn.Linear(4,1)
		self.fc1.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		if(action.dim() == 1):
			action = action[:, None]

		s1 = F.relu(self.fcs1(state))
		a1 = F.leaky_relu(self.fca1(action))

		if(a1.dim() == 1):
			A1 = a1[None,:]
			x = torch.cat((s1,A1),dim=1)
		else:
			x = torch.cat((s1,a1),dim=1)

		x = self.fc1(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,3)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(3,action_dim)
		self.fc2.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		action = F.leaky_relu(self.fc2(x))

		action = ((action/2) + 0.5) * self.action_lim 

		return action