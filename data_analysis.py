import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

all_data = pd.DataFrame()
df = pd.read_excel("Xbox 5-day auctions.xlsx")
all_data = all_data.append(df)

all_data_7 = pd.DataFrame()
df = pd.read_excel("Xbox 7-day auctions.xlsx")
all_data_7 = all_data_7.append(df)

grouped_df =  all_data.groupby(['bidder'])
max_bid = grouped_df.max()['bid']
fit_data = max_bid.tolist()
mu,std = norm.fit(fit_data)
print("mu: " + str(mu))
print("std: " + str(std))

grouped_df =  all_data_7.groupby(['bidder'])
max_bid = grouped_df.max()['bid']
fit_data = max_bid.tolist()
mu,std = norm.fit(fit_data)
print("mu: " + str(mu))
print("std: " + str(std))

"""
5 days: 
    mu: 97.09859872611464
    std: 78.35902558754839
7 days:
    mu: 95.9226981707317
    std: 64.89876178077046
"""


# n, bins, patches = plt.hist(x=max_bid, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Utility')
# plt.ylabel('Frequency')
# plt.title('Utility Distribution for Xbox Acutions')
# maxfreq = n.max()
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()