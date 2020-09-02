import numpy as np
import tensorflow as tf
import os
import datetime
from statistics import mean
import random
import log
import games as g
import dqn as dqn

# Turn on verbose logging, 0: No verbose, 1: Rough verbose, 2: Step-by-step-verbose, 3: Step-by-step-detailed-verbose
verbose = 0

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
            randMove, q = self.TrainNet.get_q(np.array(observations), epsilon) # TrainNet determines favorable action
            action = 0
            
            if not randMove:
                q_list_prob=[]
                q_list_min = np.min(q)
                q_list_max = np.max(q)
                for qi in q:
                    q_list_prob.append(float((qi-q_list_min)/(q_list_max-q_list_min)))
                for i, p in enumerate(q_list_prob):
                    if environment.isIllegalMove(i):
                        q_list_prob[i] = - 1
                action = np.argmax(q_list_prob)
                
            else:
                action = q

            prev_observations = observations # saves observations
            result = environment.step(action)
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

    def playNewModel(self, state, environment, epsilon, copy_step):
        environment.reset()
        rewards = 0
        iter = 0
        done = False
        observations = state
        losses = list()
        illegal_moves = 0
        while not done: # observes until game is done 
            # This part stays mostly the same, the change is to query a model for Q values
            
            action = self.TrainNet.get_action(np.array(state))

            prev_observations = observations # saves observations
            # Uncomment following line if you want to test how a purely random agent performs
            # action = random.randint(0,8)
            result = environment.step(action)
            observations = result[0]
            state = observations
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
            # Every step we update replay memory and train main network
            self.TrainNet.update_replay_memory((np.array(prev_observations), action, reward, np.array(observations), done))
            self.TrainNet.train(np.array(observations))
            # loss = TrainNet.train(TargetNet) # returns loss 
            # if isinstance(loss, int): # checks if loss is an integer
            #     losses.append(loss)
            # else:
            #     losses.append(loss.numpy()) # converted into an integer
            # iter += 1 # increment the counter
            # if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
            #     TargetNet.copy_weights(TrainNet) 
        return rewards, 50, won, lose, illegal_moves #returns rewards and average
    
    def playModel2(self, state, environment, epsilon, copy_step):
        environment.reset()
        rewards = 0
        iter = 0
        done = False
        observations = state
        losses = list()
        illegal_moves = 0
        while not done:
            action = self.TrainNet.act(state)

            prev_observations = observations # saves observations
            # Uncomment following line if you want to test how a purely random agent performs
            # action = random.randint(0,8)
            result = environment.step(action)
            observations = result[0]
            state = observations
            reward = result[1]
            done = result[2]
            illegalmove = result[5]

            if illegalmove and False:
                print(prev_observations)
                print("Action: ",action, ";Done: ",done)

            
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

            self.TrainNet.update_mem(prev_observations, action, reward, observations, done)
            self.TrainNet.train()

        return rewards, 50, won, lose, illegal_moves
            
    def main(self, testing):
        # Dict of all games for generalization purposes, values are:
        # 0: play_game func, 1: Which environment to use, 2: Subfolder for checkpoints, log and figures, 3: Plotting func
        games = {"tictactoe":[self.play_tictactoe,g.tictactoe,"tictactoe",log.plotTicTacToe]}
        
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
        
        self.TrainNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
        self.TargetNet = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)

        # self.TrainNet = dqn.DQNAgent()
        # self.TargetNet = dqn.DQNAgent()

        # self.TargetNet = dqn.agent()
        # self.TrainNet = dqn.agent()

        load = False
        if load:
            model_name = "model.2020.08.12-19.42.21-I.100-N.1000"
            directory = "tictactoe/models/"+model_name+"/TrainNet/"
            self.TrainNet.model = tf.saved_model.load(directory)
            directory = "tictactoe/models/"+model_name+"/TargetNet/"
            self.TargetNet.model = tf.saved_model.load(directory)
            

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
                print("Episode: {0:{1}.0f} | Episode Reward: {2:2.0f} | Eps.: {3:2.0f} | Avg. Rew. (last {4:.0f}): {5:2.3f} | Episode Loss: {6:.3f} | Wins: {7:2.0f} | Lose: {8:.0f}".format(n, len(str(N)), total_reward, epsilon, log_interval, avg_rewards, losses, win_count, lose_count))
                
                f = open(log_path, "a")
                f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(win_count))+";"+ str(lose_count)+";"+ str(illegal_moves)+"\n")
                illegal_moves = 0
                f.close()
                win_count = 0
                lose_count = 0

                # Save the models
                tf.saved_model.save(self.TrainNet.model, checkpoint_path+"/TrainNet")
                tf.saved_model.save(self.TargetNet.model, checkpoint_path+"/TargetNet")
        print("avg reward for last 100 episodes:", avg_rewards)    
        game[3](log_path)

if __name__ == '__main__':
    # Set Parameter to true if you want to load the model on the path above and test it
    train_dqn = train_dqn()
    train_dqn.main(False)



