import os
import datetime
import matplotlib.pyplot as plt

def plot(log_path):
    data=[]
    with open(log_path) as f: # open log.txt
        for line in f.readlines(): #read all lines and safe as list in line
            line=line.split(';') #separate data by ; into a list
            float_list = [float(i) for i in line] #convert into float
            data.append(float_list)  #append data
    total_reward = [] #create datalist
    n = []
    epsilon = []
    avg_reward = []
    losses = []
    win_count = []
    for i in range(len(data)): #append all data
        n.append(data[i][0])
        total_reward.append(data[i][1])
        epsilon.append(data[i][2]*1000)
        avg_reward.append(data[i][3])
        losses.append(data[i][4]/30)
        win_count.append(data[i][5]*5)
    plt.plot(n, total_reward, 'r', label="Total Reward") #plot data
    plt.plot(n, epsilon, 'g', label="Epsilon (amplified x1000)")
    plt.plot(n, avg_reward, 'b', label="Avg. Reward")
    plt.plot(n, losses, 'y', label="Losses (30)")
    plt.plot(n, win_count, 'k', label="Wins per 100 (amplified x5)")
    plt.title("Log")
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    plt.savefig('figures/fig.'+current_time+".pdf")
    plt.show()


