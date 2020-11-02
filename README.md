# 238 Final Project
Implementation to learn a bidding agent that tries to maximize its gain when attending an auction against another opponent, whose utility remains unknown and could only be inferred from the sequence of bids. In this project, our group focuses solely on auction for Xbox.
The agent learn the strategy through simulation against human-like random agent, who bids a price in between the current bid and his/hers utility, with a possibility to leave the auction if the bid price comes too close to his/hers utility. If the bid price exceeds his/hers utility, then he/his will for sure leave the auction.

Our group aims to learn an agent such that when the private utility is higher than that of the opponent, the agent will learn to bid a price that helps him win the product with lower price.

Two approaches are taken for this project - discrete and continuous ones. 

The discrete approach consults the Q-Learning algorithm and learn a Q-table by discretizing the bidding price for Xbox into $1 range, and learn the action-value function.

The continuous approach consults the PyTorch implementation of continuous action actor-critic algorithm(https://github.com/vy007vikas/PyTorch-ActorCriticRL), which utilizes DeepMind's Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971) method for updating the actor and critic networks. Our group adapts the algorithm to our setup.

## Environment
The environment serves as an interactive center to offer feedbacks to both agents. Since auction state results are dependent on both agents' actions, the environment is implemented with a time lag such that when the second agent takes an action, the return state and reward is the one for the first agent. Detail explanation can be found in our report.

For discrete approach, the reward along the auction is 0. Since our group aims to model realisitic auction scenario as much as possible, time step (measured in terms of number of times bidding) is taken into account when computing the reward. This measurement is to encourage the agent to not always add up only a small fraction amount of time, which is not considered a good bidding strategy, and also model the time committment that real individual spent on bidding a product.

For continuous case, similar idea is offered, except that due to the potential long time step, our group discounts the reward by time step.

## Approach
### Discrete
The detail implementation of the discrete algorithm is put in the __discrete.py__ python file.

We measure the discrete approach using the following two ways: first, porprotion of positive rewards(i.e. using a bid price less than actual utility and takes steps as minimum as possible) over the total number of situations that the agent's utility is actually larger than its opponent; second, the numeric reward value when the agent's utility is indeed larger than its opponent. 

The discrete approach runs for __25000__ iterations in order to converge to an optimal strategy. Below are the two plots for the abovementioned metrics.

label 1 | label 2
--- | ---
![](https://github.com/kristinayige/CS238_Final_Project/blob/main/discrete_ratio.png "Positive Reward Porprotion") | ![](https://github.com/kristinayige/CS238_Final_Project/blob/main/discrete_rewards.png "Numeric Reward Trend"){:width="100px"}

### Continuous
The detail implementation of the continuous algorithm is put in the __continuous.py__ python file.

