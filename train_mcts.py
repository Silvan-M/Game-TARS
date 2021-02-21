# Thanks to floriangardin/connect4-mcts for the main structure of the MCTS (Monte-Carlo tree search) 
from connect4 import *
from mcts import Node, train_mcts_once, train_mcts_during
from collections import defaultdict
import numpy as np
import dqn
import os
import logging
import random

# Disable TensorFlow logging:
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import datetime
from statistics import mean
import log
import dqn


def play_connectfour(mcts, TrainNet, TargetNet):
    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)
    total_reward = 0
    losses = 0
    won = False
    lose = False
    tie = 0 # Binary, don't ask why
    # test AI with real play
    grid = create_grid()
    round = 0
    training_time = 500
    node = mcts
    while True:
        if (round % 2) == 0:
            n, TrainNet, TargetNet = train_mcts_during(node, training_time, TrainNet, TargetNet)
            moves = n.get_children_moves()
            move = 0
            if moves != []:
                move = random.choice(moves)
            node = n.get_children_with_move(move)
        else:
            node, TrainNet, TargetNet = train_mcts_during(node, training_time, TrainNet, TargetNet)
            # print([(n.win, n.games) for n in node.children])
            node, move = node.select_move()

        grid, winner = play(grid, move)


        assert np.sum(node.state - grid) == 0, node.state
        if winner != 0:
            if winner == 1:
                won = True
                lose = False
                tie = 0
            else:
                won = False
                lose = True
                tie = 0
            break
        # elif winner == 0:s
        #     won = False
        #     lose = False
        #     tie = 1
        #     break
        round += 1
    
    return mcts, total_reward, losses, won, lose, tie, TrainNet, TargetNet

if __name__ == '__main__':
    # Dict of all games for generalization purposes, values are:
    # 0: play_game func, 1: Which environment to use, 2: Subfolder for checkpoints, log and figures, 3: Plotting func, 4: PlayGameReturn (0 = win&lose, 1 = points), 5: optimal log_interval
    games = {"connectfour":[play_connectfour, train_mcts_once,"connectfour",log.plotConnectFour,0,10]}
    
    game = "useInput"
    
    while game == "useInput":
        userInp = input("Which game do you want to play?\n")
        if userInp in games.keys():
            game = userInp
        elif userInp == "0":
            game = "connectfour"
        else:
            print("Not recognized input, please try again")
    print("Training "+game)
    
    # Here you can choose which of the games declared above you want to train, feel free to change!
    game = games[game]
    batch_size = 1
    gamma = 0.9
    copy_step = 50
    num_states = 84
    num_actions = 7 # 7 columns
    hidden_units = [50]*7
    max_experience = 50000
    min_experience = 100
    alpha = 0.01
    epsilon = 1
    min_epsilon = 0.05
    decay = 0.99985
    state = [0]*num_states
    # state: the initial state
    # gamma: discount factor, weights importance of future reward [0,1]
    # copy_step: the amount of episodes until the TargetNet gets updated
    # num_states: Amount of states, num_actions: Amount of actions
    # hidden_units: Amount of hidden neurons 
    # max_experiences: sets the maximum data stored as experience, if exceeded the oldest gets deleted
    # min_experiences: sets the start of the agent learning
    # batch_size: amount of data processed at once
    # alpha: learning rate, defines how drastically it changes weights
    

    TrainNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experience, min_experience, batch_size, alpha)
    TargetNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experience, min_experience, batch_size, alpha)

    # LOADING MODELS - Set one of the variables if you want to load a model
    # Define model name
    model_name = ""
    # Alternatively define relative model path
    model_path = ""
    
    if model_name != "" or model_path != "":
        if model_path == "":
            model_path = game[2]+"/models/"+model_name
        TrainNet.model = tf.saved_model.load(model_path+"/TrainNet")
        TargetNet.model = tf.saved_model.load(model_path+"/TargetNet")
        
    N = 5000
    while True:
        try:
            N = int(input("How many episodes do you want to train?\n"))
            break
        except ValueError:
            pass
        print("Invalid input!")

    total_rewards = np.empty(N)
    win_count = 0
    lose_count = 0
    log_interval = game[5]

    # For storing logs and model afterwards
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    timeAndInfo = current_time+"-I."+str(log_interval)+"-N."+str(N)
    log_path = game[2]+"/logs/log."+timeAndInfo+".txt" # Model saved at "tictactoe/logs/log.Y.m.d-H:M:S-N.amountOfEpisodes.txt"
    checkpoint_path = game[2]+"/models/model."+timeAndInfo # Model saved at "tictactoe/models/model.Y.m.d-H:M:S-N.amountOfEpisodes"
    illegal_moves = 0
    
    mcts = None

    for i in range(100):
        mcts, TrainNet, TargetNet = game[1](mcts, TrainNet, TargetNet)

    
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        mcts, total_reward, losses, won, lose, illegal_moves_game, TrainNet, TargetNet = game[0](mcts, TrainNet, TargetNet)
        if won:
            win_count += 1
        if lose:
            lose_count += 1
        total_rewards[n] = total_reward
        #print(illegal_moves_game)
        avg_rewards = total_rewards[max(0, n - log_interval):(n + 1)].mean()
        illegal_moves += illegal_moves_game
        if (n % log_interval == 0) and (n != 0) or (n == N-1):
            print("N: {0:{1}.0f} | Epsilon: {2:2.2f} | Avg. Rew. (last {3:.0f}): {4:9.3f} | Eps. Loss: {5: 10.1f} | Wins: {6:2.0f} | Lose: {7:.0f}".format(n, len(str(N)), epsilon, log_interval, avg_rewards, losses, win_count, lose_count))
            
            f = open(log_path, "a")
            f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(win_count))+";"+ str(lose_count)+";"+ str(illegal_moves)+"\n")
            illegal_moves = 0
            f.close()
            win_count = 0
            lose_count = 0

            # Save the models
            tf.saved_model.save(TrainNet.model, checkpoint_path+"/TrainNet")
            tf.saved_model.save(TargetNet.model, checkpoint_path+"/TargetNet")
    print("avg reward for last 100 episodes:", avg_rewards)    
    game[3](log_path)

