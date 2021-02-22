# %matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import pyNetLogo

netlogo = pyNetLogo.NetLogoLink(gui=True)

netlogo.load_model('./models/Wolf Sheep Predation_v6.nlogo')
netlogo.command('setup')

agent_xy = pd.read_excel('./data/xy_DataFrame.xlsx')
agent_xy[['who','xcor','ycor']].head(5)
netlogo.write_NetLogo_attriblist(agent_xy[['who','xcor','ycor']], 'a-sheep')

if netlogo.netlogo_version == '6':
    x = netlogo.report('map [s -> [xcor] of s] sort sheep')
    y = netlogo.report('map [s -> [ycor] of s] sort sheep')
elif netlogo.netlogo_version == '5':
    x = netlogo.report('map [[xcor] of ?1] sort sheep')
    y = netlogo.report('map [[ycor] of ?1] sort sheep')

fig, ax = plt.subplots(1)

ax.scatter(x, y, s=4)
ax.set_xlabel('xcor')
ax.set_ylabel('ycor')
ax.set_aspect('equal')
fig.set_size_inches(5,5)

plt.show()