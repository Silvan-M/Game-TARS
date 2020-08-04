import os
from datetime import date 
def plot():
    data=[]
    with open('log.txt') as f:
        for line in f.readlines():
            line=line.split(';')
            int_list = [int(i) for i in line]
            data.append(int_list)  
    print(data)
    total_reward = []
    n = []
    epsilon = []
    avg_reward = []
    losses = []
    for i in range(len(data)):
        n.append(data[i][0])
        total_reward.append(data[i][1])
        epsilon.append(data[i][2])
        avg_reward.append(data[i][3])
        losses.append(data[i][4])
    print(total_reward)
    print(n)
    print(epsilon)
    print(avg_reward)
    print(losses)

f = open("log.txt", "w")
n=2
total_reward=2
epsilon=3
avg_rewards=8
losses =6
f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str( losses))+"\n")
f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str( losses))+"\n")
f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str( losses))+"\n")
f.close()

plot()
