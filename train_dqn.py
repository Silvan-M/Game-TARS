import numpy as np
import tensorflow as tf
import os
import datetime
from statistics import mean
import random
import log
import games as g
import dqn as dqn

def play_tictactoe(state, environment, TrainNet, TargetNet, epsilon, copy_step):
    environment.reset()
    rewards = 0
    iter = 0
    done = False
    observations = state
    losses = list()
    illegal_moves = 0
    while not done: # observes until game is done 
        action = TrainNet.get_action(observations, epsilon) # TrainNet determines favorable action
        prev_observations = observations # saves observations
        result = environment.step_random(action)
        observations = result[0]
        reward = result[1]
        done = result[2]
        illegalmove = result[5]
        if illegalmove:
            illegal_moves += 1
        if result[3] == 1:
            won = True
        else:
            won = False
        if result[4] == 1:
            lose = True
        else:
            lose = False
        rewards += reward        
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done} # make memory callable as a dictionary
        TrainNet.add_experience(exp)# memorizes experience, if the max amount is exceeded the oldest element gets deleted
        loss = TrainNet.train(TargetNet) # returns loss 
        if isinstance(loss, int): # checks if loss is an integer
            losses.append(loss)
        else:
            losses.append(loss.numpy()) # converted into an integer
        iter += 1 # increment the counter
        if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
            TargetNet.copy_weights(TrainNet) 
    return rewards, mean(losses), won, lose, illegal_moves #returns rewards and average

def main():
    # Dict of all games for generalization purposes, values are:
    # 0: play_game func, 1: Which environment to use, 2: Subfolder for checkpoints, log and figures, 3: Plotting func
    games = {"tictactoe":[play_tictactoe,g.tictactoe,"tictactoe",log.plotTicTacToe]}
    
    # Here you can choose which of the games declared above you want to train, feel free to change!
    game = games["tictactoe"]

    environment = game[1]()
    state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = environment.variables
    # state: the initial state
    # gamma: discount factor, weights importance of future reward [0,1]
    # copy_step: the amount of episodes until the TargetNet gets updated
    # num_states: Amount of states, num_actions: Amount of actions
    # hidden_units: Amount of hidden neurons 
    # max_experiences: sets the maximum data stored as experience, if exceeded the oldest gets deleted
    # min_experiences: sets the start of the agent learning
    # batch_size: amount of data processed at once
    # alpha: learning rate, defines how drastically it changes weights
    
    TrainNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
    TargetNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)

    N = int(input("How many episodes do you want to train?\n"))
    total_rewards = np.empty(N)
    win_count = 0
    lose_count = 0
    log_interval = 100

    # For storing logs and model afterwards
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    timeAndInfo = current_time+"-I."+str(log_interval)+"-N."+str(N)
    log_path = game[2]+"/logs/log."+timeAndInfo+".txt" # Model saved at "tictactoe/logs/log.Y.m.d-H:M:S-N.amountOfEpisodes.txt"
    checkpoint_path = game[2]+"/models/model."+timeAndInfo # Model saved at "tictactoe/models/model.Y.m.d-H:M:S-N.amountOfEpisodes"
    illegal_moves = 0
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses, won, lose, illegal_moves_game = game[0](state, environment, TrainNet, TargetNet, epsilon, copy_step)
        if won:
            win_count += 1
        if lose:
            lose_count += 1
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - log_interval):(n + 1)].mean()
        illegal_moves += illegal_moves_game
        if (n % log_interval == 0) and (n != 0) or (n == N-1):
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last "+str(log_interval)+"):", avg_rewards,
                  "episode loss: ", losses, "wins: ",win_count, "lose: ", lose_count, "illegal moves: ",illegal_moves)
            f = open(log_path, "a")
            illegal_moves = 0
            f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(win_count))+";"+ str(lose_count)+"\n")
            f.close()
            win_count = 0
            lose_count = 0

            # Save the models
            tf.saved_model.save(TrainNet.model, checkpoint_path+"/TrainNet")
            tf.saved_model.save(TargetNet.model, checkpoint_path+"/TargetNet")
    print("avg reward for last 100 episodes:", avg_rewards)    
    game[3](log_path)

if __name__ == '__main__':
    main()



