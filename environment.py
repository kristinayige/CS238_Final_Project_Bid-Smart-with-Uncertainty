from scipy.stats import truncnorm
from scipy.stats import uniform

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
    def __init__(self, u1, u2, p, t=None):
        self.p = p
        self.u1 = u1
        self.u2 = u2
        self.type = t
        self.bid1 = 0    # initialize current bid
        self.bid2 = 0
        self.status = True

    # this step function has a time lag: when action is input, it will
    # return the reward for the opponent and the new observation
    # in order to compute reward, need to also record previous action of
    # the opponent
    def step(self, act1, act2, idx=1):
        # if status is done, 0 reward and ceased
        if not self.status:
            obs = ([1, self.bid1, self.bid2])
            reward = -100
            done = True
            return (obs, reward, done)
        # otherwise
        if self.type is None:
            return self.interact_update(act1, act2, idx)
        else:
            return self.update(act1)

    def update(self, act1):
        if act1 < self.bid2:
            # treat as leave
            obs = ([1, self.bid1, self.bid2])
            reward = -100
            done = True
            return (obs, reward, done)
        self.bid1 = act1
        # since in this case only one agent, we stochastically smapled agent 2's bid
        act2 = uniform.rvs() * (self.u2 - self.bid2) + self.bid2
        if act2 < self.bid1:
            # if bidder 2 lose
            obs = ([1, self.bid1, self.bid2])
            reward = self.u1 - self.bid1
            done = True
            return (obs, reward, done)
        self.bid2 = act2
        obs = ([0, self.bid1, self.bid2])
        prob_win = 1 - (truncnorm.cdf(self.bid2, self.u1) / truncnorm.cdf(self.bid2, self.p))
        if prob_win > 0.5:
            reward = prob_win * (self.u1 - self.bid2)
        else:
            reward = (1 - prob_win) * -100

        return (obs, reward, False)

    def interact_update(self, act1, act2, idx=1):
        if idx == 1:
            # if is the first player, return result for agent 2
            if act1 < self.bid2:  
                # if do not outplay opponent, treat as leave auction
                obs = ([1, self.bid1, self.bid2])
                reward = self.u2 - self.bid2
                self.status = False # end the auction
                # we need to also update the result to the first
                # agent, so not done yet
                return (obs, reward, False)
            else:
                self.bid1 = act1
                obs = ([0, self.bid1, self.bid2])
                reward = -1 # as game not ended yet, negative reward for time spent
                return (obs, reward, False)
        else:
            if act2 < self.bid1:
                obs = ([1, self.bid1, self.bid2])
                reward = self.u1 - self.bid1
                self.status = False # end the auction
                # we need to also update the result to the first
                # agent, so not done yet
                return (obs, reward, False)
            else:
                self.bid2 = act2
                obs = ([0, self.bid1, self.bid2])
                reward = -1 # as game not ended yet, negative reward for time spent
                # prob_win = truncnorm.cdf(self.bid)
                return (obs, reward, False)

    def reset(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        self.bid1 = 0 # reset environment
        self.bid2 = 0
        self.status = True

        # observatin contains three component: (my_bid_value, opponent_bid_value, status)
        return ([0, self.bid1, self.bid2])
