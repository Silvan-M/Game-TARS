import os
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as nps
import numpy as np

def plotTicTacToe(log_path):
    # Filename is used to create 
    data=[]
    timeAndInfo = log_path[19:-4]
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
    lose_count = []
    tie_count = []
    illegal_move_count = []
    N_start_index = log_path.find("-N.")
    N = int(log_path[N_start_index+3:-4])
    log_interval_start_index = log_path.find("-I.")
    log_interval = int(log_path[log_interval_start_index+3:N_start_index])

    for i in range(len(data)): #append all data
        n.append(data[i][0])
        total_reward.append(data[i][1])
        epsilon.append(data[i][2]*1000)
        avg_reward.append(data[i][3])
        losses.append(data[i][4]/30)
        win_count.append(data[i][5])
        lose_count.append(data[i][6])
        tie_count.append(log_interval-data[i][5]-data[i][6])
        illegal_move_count.append(data[i][7])
    #plt.plot(n, total_reward, 'r', label="Total Reward") #plot data
    #plt.plot(n, epsilon, 'g', label="Epsilon (amplified x1000)")
    #plt.plot(n, avg_reward, 'b', label="Avg. Reward")
    #plt.plot(n, lose, 'y', label="Lose (30)")

    plt.figure(0)
    plotIllegalMove = False
    if plotIllegalMove:
        plt.plot(n, illegal_move_count, 'k', label="Illegal moves per "+str(log_interval))
        plt.title("Log N"+str(N))
        plt.xlabel("Illegal Moves")
        plt.ylabel("Value")
        plt.legend(loc="upper right")
        plt.savefig('tictactoe/figures/fig.'+timeAndInfo+"ILLEGALMOVES.pdf")

        plt.figure(1)
    plt.plot(n, win_count, 'g', label="Wins per "+str(log_interval))
    plt.plot(n, lose_count, 'r', label="Losses per "+str(log_interval))
    plt.plot(n, tie_count, 'k', label="Ties per "+str(log_interval))
    plt.title("Log N"+str(N))
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.legend(loc="upper right")
    z = nps.polyfit(range(len(win_count)), win_count, 1)
    p = np.poly1d(win_count)
    x = np.arange(10)
    y = p(x)
    plt.plot(x,y)
    plt.savefig('tictactoe/figures/fig.'+timeAndInfo+".pdf")
    plt.show()