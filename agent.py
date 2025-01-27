from scipy.stats import truncnorm
from scipy.stats import uniform

class Agent:
    def __init__(self, p=300):
        a = (50-95)/64
        b = (150-95)/64
        util = truncnorm.rvs(a, b)
        self.utility = util * 64 + 95

    def action(self, curr_bid):
        if self.utility <= curr_bid + (uniform.rvs() * 10 - 5):
            return 0
        else:
            return uniform.rvs() * (self.utility - curr_bid) + curr_bid