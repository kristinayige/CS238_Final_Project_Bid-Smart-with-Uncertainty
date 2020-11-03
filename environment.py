from scipy.stats import truncnorm
from scipy.stats import uniform
import math

class Environment:
    """Environment holds the auction with two participants, each participant
    holds a private value about the product with price p
    Based on our utilization, price p remains unchanged, but the utility
    can be changed when the environment is reset, simulating a new episode
    The agent assume one participant's point of view and try to learn
    strategy against opponent whose private value is unknown with prior
    that the utility is distributed as a truncated normal distribution

    :param u1, u2: true utility privately held by each player, used to determine
                   the reward for each player when the auction ends
    :param p:      true price of the prodcut being bidded against, used to restrict
                   the reward range
    :param t:      the type of environment one is in, there are two types of environment
                    - None:         when the reward is only given after one episode of 
                                    auction is completed with time step -1
                    - Trun_Gauss:   when the reward is given as above, except
                                    using Truncated Gaussian
    """
    def __init__(self, p=500, u1=100, u2=50, t=0):
        self.p = p
        self.u1 = u1
        self.u2 = u2
        self.type = t
        self.bid1 = 0    # initialize current bid
        self.bid2 = 0
        self.time = 0
        self.status = True

    def set_utility(self, u1, u2):
        self.u1 = u1
        self.u2 = u2

    def step(self, act1, act2, idx=1):
        # if status is done, 0 reward and ceased
        if not self.status:
            obs = ([self.time, max(self.bid1, self.bid2)])
            reward = -10
            done = True
            return (obs, reward, done, max(self.bid1, self.bid2))
        # otherwise
        return self.interact_update(act1, act2, idx)

    def interact_update(self, act1, act2, idx=1):
        if idx == 1:
            # increment the time only if our own agent plays
            self.time += 1
            # if is the first player, return result for agent 2
            if act1 <= self.bid2:
                obs = ([self.time, self.bid2])
                if self.type == 1:
                    reward = math.pow(0.99, self.time) * (self.u2 - self.bid2)
                else:
                    reward = (self.u2 - self.bid2)
                self.status = False
                return (obs, reward, True, max(self.bid1, self.bid2))
            else:
                self.bid1 = act1
                obs = ([0, self.bid1, self.bid2])
                reward = -1
                return (obs, reward, False, max(self.bid1, self.bid2))
        else:
            if act2 <= self.bid1:
                obs = ([self.time, self.bid1])
                if self.type == 1:
                    reward = math.pow(0.99, self.time) * (self.u1 - self.bid1)
                else:
                    reward = (self.u1 - self.bid1)
                self.status = False
                return (obs, reward, True, max(self.bid1, self.bid2))
            else:
                self.bid2 = act2
                obs = ([self.time, self.bid2])
                reward = -1
                return (obs, reward, False, max(self.bid1, self.bid2))

    def reset(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        self.bid1 = 0 # reset environment
        self.bid2 = 0
        self.status = True
        self.time = 0

        # observatin contains three component: (my_bid_value, opponent_bid_value, status)
        return ([self.time, 0], 0)
