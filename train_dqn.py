
import numpy as np
import os
import logging

# Disable TensorFlow logging:
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import datetime
from statistics import mean
import random
import log
import games as g
import dqn as dqn
global MMA
MMA = True # True = Random, MinMaxAlg = False
# Turn on verbose logging, 0: No verbose, 1: Rough verbose, 2: Step-by-step-verbose, 3: Step-by-step-detailed-verbose
verbose = 1

class train_dqn():
    def play_tictactoe(self, state, environment, epsilon, copy_step):
        environment.reset()
        rewards = 0
        iter = 0
        done = False
        observations = environment.convert0neHot(state)
        losses = list()
        illegal_moves = 0
        while not done: # observes until game is done 
            action = 0

            # Set to False if you want illegalmoves, if True it will pick the highest legal q value 
            if True:
                environment.reward_illegal_move = 0
                randMove, prob = self.TrainNet.get_prob(np.array(observations), epsilon) # TrainNet determines favorable action
                
                if not randMove:
                    for i, p in enumerate(prob):
                        if environment.isIllegalMove(i):
                            prob[i] = - 1
                    action = np.argmax(prob)
                    
                else:
                    action = prob
            else:
                action = self.TrainNet.get_action(np.array(observations), epsilon) 

            prev_observations = observations # saves observations
            result = environment.step(action, MMA)
            observations = environment.convert0neHot(result[0])
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
            exp = {'s': np.array(prev_observations), 'a': action, 'r': reward, 's2': np.array(observations), 'done': done} # make memory callable as a dictionary
            self.TrainNet.add_experience(exp)# memorizes experience, if the max amount is exceeded the oldest element gets deleted
            loss = self.TrainNet.train(self.TargetNet) # returns loss 
            if isinstance(loss, int): # checks if loss is an integer
                losses.append(loss)
            else:
                losses.append(loss.numpy()) # converted into an integer
            iter += 1 # increment the counter
            if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
                self.TargetNet.copy_weights(self.TrainNet) 

            if verbose == 1:
                if done:
                    print("Reward: {0: 3.1f} | Won: {1:5} | Lose: {2:5} | Done: {3}".format(rewards,str(won),str(lose),str(done)))
            elif verbose == 2:
                print("Reward: {0: 3.1f} | Won: {1:5} | Lose: {2:5} | Done: {3}".format(rewards,str(won),str(lose),str(done)))
            elif verbose == 3:
                print(environment.state[0:3], "   ", [0,1,2])
                print(environment.state[3:6], "   ", [3,4,5])
                print(environment.state[6:9], "   ", [6,7,8])
                print("Reward: {0: 3.1f} | Won: {1:5} | Lose: {2:5} | Done: {3}\n".format(rewards,str(won),str(lose),str(done)))
        return rewards, mean(losses), won, lose, illegal_moves #returns rewards and average
    
    def play_snake(self, state, environment, epsilon, copy_step):
        environment.reset()
        rewards = 0
        apples = 0
        iter = 0
        done = False
        observations = state
        prev_observations = observations
        losses = list()
        while not done: # observes until game is done 
            
            inp = observations
            if self.TrainNet.batch_size > 1:
                # Simulate batch size of 2
                inp = [prev_observations, observations]

            _, prob = self.TrainNet.get_prob(np.array(inp), 0) # TrainNet determines favorable action
            
            if np.random.random() < epsilon:
                delete = np.argmax(prob)
                prob[delete] = -1
            
            action = np.argmax(prob)
            
            prev_observations = observations # saves observations
            done, reward, observations =  environment.step(action)
            if reward == environment.reward_apple:
                apples += 1

            rewards += reward        
            exp = {'s': np.array(prev_observations), 'a': action, 'r': reward, 's2': np.array(observations), 'done': done} # make memory callable as a dictionary
            self.TrainNet.add_experience(exp) # memorizes experience, if the max amount is exceeded the oldest element gets deleted
            loss = self.TrainNet.train(self.TargetNet) # returns loss 
            if isinstance(loss, int): # checks if loss is an integer
                losses.append(loss)
            else:
                losses.append(loss.numpy()) # converted into an integer
            iter += 1 # increment the counter
            if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
                self.TargetNet.copy_weights(self.TrainNet) 

            if verbose == 1:
                if done:
                    print("Reward: {0: 3.1f} | Apples: {1:5} | Done: {2}".format(rewards,str(apples),str(done)))
            elif verbose == 2:
                print("Reward: {0: 3.1f} | Apples: {1:5} | Done: {2}".format(rewards,str(apples),str(done)))
            elif verbose == 3:
                for row in range(0, environment.field_size):
                    print(environment.field[(row*environment.field_size):(row*environment.field_size+environment.field_size)])
                print("Reward: {0: 3.1f} | Apples: {1:5} | Done: {2}\n".format(rewards,str(apples),str(done)))
        return rewards, mean(losses), apples #returns rewards and average
    
    def play_space_invader(self, state, _, epsilon, copy_step):
        check_action = None
        check_action_count = 0
        environment = g.space_invader()
        rewards = 0
        iter = 0
        done = False
        observations = np.float32(np.asarray([0]*self.num_states))
        #observations = np.float32(np.asarray(environment.replacer()).flatten())
        prev_observations = observations
        losses = list()
        nr = 0
        while not done: # observes until game is done 
            
            inp = observations
            if self.TrainNet.batch_size > 1:
                # Simulate batch size of 2
                inp = [prev_observations, observations]
            nr += 1

            action = self.TrainNet.get_action(np.array(inp), 0) # TrainNet determines favorable action
            convAction = ['N', False]
            if action == 0:
                convAction = ['L', False]
            elif action == 1:
                convAction = ['R', False]
            elif action == 2:
                convAction = ['N', True]
            if check_action == convAction: 
                check_action_count += 1
            else:
                check_action = convAction 
            if check_action_count > 500:
                reward -= 10000
                check_action_count = 0
                if verbose > 1:
                    print('killed by nothingness',convAction)

                done = True
            prev_observations = observations # saves observations
            reward, observations = environment.step(convAction)

            observations = np.asarray(observations).flatten()

            if environment.health <= 0:
                done = True
                reward = environment.reward_ship_destroyed

      
            rewards += reward
            exp = {'s': np.array(prev_observations), 'a': action, 'r': reward, 's2': np.array(observations), 'done': done} # make memory callable as a dictionary
            self.TrainNet.add_experience(exp) # memorizes experience, if the max amount is exceeded the oldest element gets deleted
            loss = self.TrainNet.train(self.TargetNet) # returns loss 
            if isinstance(loss, int): # checks if loss is an integer
                losses.append(loss)
            else:
                losses.append(loss.numpy()) # converted into an integer
            iter += 1 # increment the counter
            if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
                self.TargetNet.copy_weights(self.TrainNet) 

            if verbose == 1:
                if done:
                    print("Reward: {0: 3.1f} | Score: {1:5} | Done: {2}".format(rewards,str(environment.score[3]),str(done)))
            elif verbose == 2:
                print("Reward: {0: 3.1f} | Score: {1:5} | Done: {2} | Action: {2}".format(rewards,str(environment.score[3]),str(done)),str(action))
            elif verbose == 3:
                print("Reward: {0: 3.1f} | Score: {1:5} | Done: {2} | Action: {2}".format(rewards,str(environment.score[3]),str(done)),str(action))
        return rewards, mean(losses), environment.score[3] #returns rewards and average

            
    def main(self, testing):
        # Dict of all games for generalization purposes, values are:
        # 0: play_game func, 1: Which environment to use, 2: Subfolder for checkpoints, log and figures, 3: Plotting func, 4: PlayGameReturn (0 = win&lose, 1 = points), 5: optimal log_interval
        games = {"tictactoe":[self.play_tictactoe,g.tictactoe,"tictactoe",log.plotTicTacToe,0,100],"snake":[self.play_snake,g.snake,"snake",log.plotSnake,1,10],"spaceinvaders":[self.play_space_invader,g.space_invader,"spaceinvader",log.plotSpaceInvader,1,10]}
        
        # Here you can choose which of the games declared above you want to train, feel free to change!
        game = games["spaceinvaders"]

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
        self.num_states = num_states
        
        self.TrainNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
        self.TargetNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)

        # LOADING MODELS - Set one of the variables if you want to load a model
        # Define model name
        model_name = ""
        # Alternatively define relative model path
        model_path = ""
        
        if model_name != "" or model_path != "":
            if model_path == "":
                model_path = game[2]+"/models/"+model_name
            self.TrainNet.model = tf.saved_model.load(model_path+"/TrainNet")
            self.TargetNet.model = tf.saved_model.load(model_path+"/TargetNet")
            

        N = int(input("How many episodes do you want to train?\n"))
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

        # If output is win&lose
        if game[4] == 0:
            for n in range(N):
                epsilon = max(min_epsilon, epsilon * decay)
                total_reward, losses, won, lose, illegal_moves_game = game[0](state, environment, epsilon, copy_step)
                if won:
                    win_count += 1
                if lose:
                    lose_count += 1
                total_rewards[n] = total_reward
                #print(illegal_moves_game)
                avg_rewards = total_rewards[max(0, n - log_interval):(n + 1)].mean()
                illegal_moves += illegal_moves_game
                if (n % log_interval == 0) and (n != 0) or (n == N-1):
                    print("Eps.: {0:{1}.0f} | Eps. Rew.: {2: 4.0f} | Epsilon: {3:2.0f} | Avg. Rew. (last {4:.0f}): {5:2.3f} | Eps. Loss: {6: 10.1f} | Wins: {7:2.0f} | Lose: {8:.0f}".format(n, len(str(N)), total_reward, epsilon, log_interval, avg_rewards, losses, win_count, lose_count))
                    
                    f = open(log_path, "a")
                    f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(win_count))+";"+ str(lose_count)+";"+ str(illegal_moves)+"\n")
                    illegal_moves = 0
                    f.close()
                    win_count = 0
                    lose_count = 0

                    # Save the models
                    tf.saved_model.save(self.TrainNet.model, checkpoint_path+"/TrainNet")
                    tf.saved_model.save(self.TargetNet.model, checkpoint_path+"/TargetNet")
        elif game[4] == 1:
            total_points = []
            for n in range(N):
                epsilon = max(min_epsilon, epsilon * decay)
                total_reward,losses, points = game[0](state, environment, epsilon, copy_step)
                total_rewards[n] = total_reward
                avg_rewards = total_rewards[max(0, n - log_interval):(n + 1)].mean()
                total_points.append(points)
                if (n % log_interval == 0) and (n != 0) or (n == N-1):
                    avg_points = sum(total_points) / len(total_points)
                    print("Eps.: {0:{1}.0f} | Eps. Reward: {2:7.0f} | Epsilon: {3:5.3f} | Avg. Rew. (last {4:.0f}): {5:6.1f} | Eps. Loss: {6:8.1f} | Points: {7:6.1f}".format(n, len(str(N)), total_reward, epsilon, log_interval, avg_rewards, losses, avg_points))
                    
                    f = open(log_path, "a")
                    f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(avg_points)+"\n"))
                    total_points = []
                    f.close()

                    # Save the models
                    tf.saved_model.save(self.TrainNet.model, checkpoint_path+"/TrainNet")
                    tf.saved_model.save(self.TargetNet.model, checkpoint_path+"/TargetNet")
        print("avg reward for last 100 episodes:", avg_rewards)    
        game[3](log_path)

if __name__ == '__main__':
    # Set Parameter to true if you want to load the model on the path above and test it
    train_dqn = train_dqn()
    train_dqn.main(False)
