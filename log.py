import os
from datetime import date 
import matplotlib.pyplot as plt

def plot():
    with open('log.txt') as f:
        next(f)
    data = f.readlines().split(';')
    data = [x.strip() for x in content] 
    x=(list(range(0, len(data))),data)
    plt.plot(x)
    plt.show()

f = open("log.txt", "w")
#f.write((n, ";", total_reward, ";", epsilon, ";", avg_rewards,";", losses))
f.close()