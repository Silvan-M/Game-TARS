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

    # intvl defines how many values should be taken the average of, this will prevent too much data points in the graph, so by default it is set to 1, but if N gets higher than 100K it will average every 100 values
    intvl = 1

    if len(data)*log_interval >= 100000:
        intvl = 100
        print("Automatically swichted to avg. every 100th value")
    elif len(data)*log_interval >= 50000:
        intvl = 10
        print("Automatically swichted to avg. every 10th value")
    intvl = 1
    amount_datapoints = 8
    avg_data = [0]*amount_datapoints

    for i in range(len(data)): #append all data
        for k in range(amount_datapoints):
            avg_data[k] += data[i][k]
        if ((i % intvl == 0) and i != 0) or (intvl == 1):
            for k, v in enumerate(avg_data):
                avg_data[k] = v/intvl

            n.append(avg_data[0])
            total_reward.append(avg_data[1])
            epsilon.append(avg_data[2]*1000)
            avg_reward.append(avg_data[3])
            losses.append(avg_data[4]/30)
            win_count.append(avg_data[5])
            lose_count.append(avg_data[6])
            tie_count.append(log_interval-avg_data[5]-avg_data[6])
            illegal_move_count.append(avg_data[7])
            avg_data = [0]*amount_datapoints
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
    
    dim = 1
    y = win_count
    coef = np.polyfit(n,y,dim)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(n, poly1d_fn(n), '--g', label = "Avg. Incr. = "+str(round(coef[0]*len(n)*100,3))+"%")

    y = lose_count
    coef = np.polyfit(n,y,dim)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(n, poly1d_fn(n), '--r', label = "Avg. Incr. = "+str(round(coef[0]*len(n)*100,3))+"%")
    
    y = tie_count
    coef = np.polyfit(n,y,dim)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(n, poly1d_fn(n), '--k', label = "Avg. Incr. = "+str(round(coef[0]*len(n)*100,3))+"%")

    # Print Averages
    def avg(lst): 
        return round(sum(lst) / len(lst),3)

    print("Avg. Reward: ",avg(total_reward),"| Avg. win: ",avg(win_count),"| Avg. lose: ",avg(lose_count),"| Avg. ties: ",avg(tie_count),"| Avg. Illegal Moves: ",avg(illegal_move_count))

    plt.legend(loc="upper right",fontsize = 'x-small')
    plt.savefig('tictactoe/figures/fig.'+timeAndInfo+".pdf")
    plt.show()

def plotSnake(log_path):
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
    pts_count = []
    N_start_index = log_path.find("-N.")
    N = int(log_path[N_start_index+3:-4])
    log_interval_start_index = log_path.find("-I.")
    log_interval = int(log_path[log_interval_start_index+3:N_start_index])

    # intvl defines how many values should be taken the average of, this will prevent too much data points in the graph, so by default it is set to 1, but if N gets higher than 100K it will average every 100 values
    intvl = 1

    if len(data)*log_interval >= 100000:
        intvl = 100
        print("Automatically swichted to avg. every 100th value")
    elif len(data)*log_interval >= 50000:
        intvl = 10
        print("Automatically swichted to avg. every 10th value")
    
    amount_datapoints = 6
    avg_data = [0]*amount_datapoints

    for i in range(len(data)): #append all data
        for k in range(amount_datapoints):
            avg_data[k] += data[i][k]
        if ((i % intvl == 0) and i != 0) or (intvl == 1):
            for k, v in enumerate(avg_data):
                avg_data[k] = v/intvl

            n.append(avg_data[0])
            total_reward.append(avg_data[1])
            epsilon.append(avg_data[2]*1000)
            avg_reward.append(avg_data[3])
            losses.append(avg_data[4]/30)
            pts_count.append(avg_data[5])
            avg_data = [0]*amount_datapoints
    #plt.plot(n, total_reward, 'r', label="Total Reward") #plot data
    #plt.plot(n, epsilon, 'g', label="Epsilon (amplified x1000)")
    #plt.plot(n, avg_reward, 'b', label="Avg. Reward")
    #plt.plot(n, lose, 'y', label="Lose (30)")

    plt.plot(n, pts_count, 'g', label="Wins per "+str(log_interval))
    plt.title("Log N"+str(N))
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    
    dim = 1
    y = pts_count
    coef = np.polyfit(n,y,dim)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(n, poly1d_fn(n), '--g', label = "Avg. Incr. = "+str(round(coef[0]*len(n)*100,3))+"%")

    # Print Averages
    def avg(lst): 
        return round(sum(lst) / len(lst),3)

    print("Avg. Reward: ",avg(total_reward),"| Avg. pts: ",avg(pts_count))
    plt.legend(loc="upper right",fontsize = 'x-small')
    plt.savefig('snake/figures/fig.'+timeAndInfo+".pdf")
    plt.show()

def plotSpaceInvader(log_path):
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
    pts_count = []
    N_start_index = log_path.find("-N.")
    N = int(log_path[N_start_index+3:-4])
    log_interval_start_index = log_path.find("-I.")
    log_interval = int(log_path[log_interval_start_index+3:N_start_index])

    # intvl defines how many values should be taken the average of, this will prevent too much data points in the graph, so by default it is set to 1, but if N gets higher than 100K it will average every 100 values
    intvl = 1

    if len(data)*log_interval >= 100000:
        intvl = 100
        print("Automatically swichted to avg. every 100th value")
    elif len(data)*log_interval >= 50000:
        intvl = 10
        print("Automatically swichted to avg. every 10th value")
    
    amount_datapoints = 6
    avg_data = [0]*amount_datapoints

    for i in range(len(data)): #append all data
        for k in range(amount_datapoints):
            avg_data[k] += data[i][k]
        if ((i % intvl == 0) and i != 0) or (intvl == 1):
            for k, v in enumerate(avg_data):
                avg_data[k] = v/intvl

            n.append(avg_data[0])
            total_reward.append(avg_data[1])
            epsilon.append(avg_data[2]*1000)
            avg_reward.append(avg_data[3])
            losses.append(avg_data[4]/30)
            pts_count.append(avg_data[5])
            avg_data = [0]*amount_datapoints
    #plt.plot(n, total_reward, 'r', label="Total Reward") #plot data
    #plt.plot(n, epsilon, 'g', label="Epsilon (amplified x1000)")
    #plt.plot(n, avg_reward, 'b', label="Avg. Reward")
    #plt.plot(n, lose, 'y', label="Lose (30)")

    plt.plot(n, pts_count, 'g', label="Score per "+str(log_interval))
    plt.title("Log N"+str(N))
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    
    dim = 1
    y = pts_count
    coef = np.polyfit(n,y,dim)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(n, poly1d_fn(n), '--g', label = "Avg. Incr. = "+str(round(coef[0]*len(n)*100,3))+"%")

    # Print Averages
    def avg(lst): 
        return round(sum(lst) / len(lst),3)

    print("Avg. Reward: ",avg(total_reward),"| Avg. pts: ",avg(pts_count))
    plt.legend(loc="upper right",fontsize = 'x-small')
    plt.savefig('spaceinvader/figures/fig.'+timeAndInfo+".pdf")
    plt.show()


# PLOT A LOG MANUALLY
# If you want to plot a Log set the model name or relative path here, if empty nothing will be plotted
model_name = {"tictactoe":"", "snake":""}
# If you want to plot your last training, change those variables
plot_last = {"tictactoe":False, "snake":False}


plot_functions = {"tictactoe": plotTicTacToe, "snake": plotSnake}
for i in model_name:
    path = model_name[i]
    if path != "":
        # Correct if user inserted a relative path
        if i+"/logs/" in path:
            path = path.replace(i+"/logs/","",1)
            print("Changing removing relative log path from string")
        if i+"/models/" in path:
            path = path.replace(i+"/models/","",1)
            print("Changing removing relative model path from string")
        # Correct if user inserted a model path
        if "model" in path:
            print("Changing 'model' to 'log' in path")
            path = path.replace("model","log",1)+".txt"
        path = i+"/logs/"+path
        print("Plotting: "+path)
        plot_functions[i](path)

for i in plot_last:
    if plot_last[i]:
        logs = os.listdir(r'{}/logs'.format(i))
        MatLogs = []
        for j, v in enumerate(logs):
            if v != ".DS_Store":
                MatLogs.append(v)
        MatLogs = sorted(MatLogs)
        path = i+"/logs/"+str(MatLogs[-1])
        print("Plotting: "+path)
        plot_functions[i](path)
