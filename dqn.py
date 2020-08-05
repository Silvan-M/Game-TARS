import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
import random
import log
import games as g

class MyModel(tf.keras.Model): # class with format tensorflow.keras.model, has ability to group layers into an object with training and inference (interpreter) features.
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__() # adds model as subclass to add the ability to differ the behaviour in training and interference
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,)) # define inputlayers, shape = determines dimension of vector [tuple]
        self.hidden_layers = [] # clears the list of hidden layers
        for i in hidden_units: 
            self.hidden_layers.append(tf.keras.layers.Dense( 
                i, activation='tanh', kernel_initializer='RandomNormal')) 
                # activation = determines the output of a node with a specific input ['tanh','sigmoid']
                # use_bias = uses bias vector [boolean] to add a value
                # kernel_initializer = initialize random weights
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function 
    # constructs a callable that executes a tensorflow graph, which is a data structures that contain a set of tf.operation objects, which reoresent units of computation, can b restoredd even without the code
    def call(self, inputs):
        z = self.input_layer(inputs) # safe input layers with contained data as z
        for layer in self.hidden_layers: 
            z = layer(z) # iterates through all hidden layers with the previous output as input
        output = self.output_layer(z) 
        return output # returns processed data of last layer


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha):
        self.num_actions = num_actions # 
        self.batch_size = batch_size # amount of data processed at once
        self.optimizer = tf.optimizers.Adam(alpha) # adjusts the weight to minimize the loss function, ADAM uses momentum and bias correction
        self.gamma = gamma #discount factor, weights importance of future reward [0,1]
        self.model = MyModel(num_states, hidden_units, num_actions) # forwards to tf.keras.model
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # memorizes initial state, actions, rewards, achieved state and done [boolean]
        self.max_experiences = max_experiences # sets the maximum data stored as experience, if exceeded the oldest gets deleted
        self.min_experiences = min_experiences # sets the start of the agent learning

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32'))) # returns a prediction by a model, inputs get converted to float32 and forwarded to tf.keras.model

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences: # checks if the amount of memorized initial states is smaller than the needed quantity
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size) # creates a random number with max length of the stored memory
        # loads the random state with coresponding actions, rewards, achieved stated and done
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1) # returns the max value of TargetNet achieved by the action
        # TargetNet = a preserved copy of the dqn which the weights get updated after a specific amount of time, is used to calculate the favorability of the action taken
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next) # replaces the reward values of a successfull action with values which take the favorability of future states into account multiplied by the discount factor.

        with tf.GradientTape() as tape: # record operation for automatic differentiation 
            selected_action_values = tf.math.reduce_sum( # returns the reduced tensor along the x-axis by adding the rows together
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1) 
                # returns predictions of the states choosen
                # one hot encoding = convertion of enumerated categories to a binary matrix which stores the range of actions
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values)) # calculates the squared difference between actual values and selected action values 
        variables = self.model.trainable_variables # calls the weights of the model 
        gradients = tape.gradient(loss, variables) # we dont know
        self.optimizer.apply_gradients(zip(gradients, variables)) #we dont know
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon: # compares a random number with the exploration factor which gets reduced over time to increase exploitation
            return np.random.choice(self.num_actions) # selects a random choice (exploration)
        else:
            a = self.predict(np.atleast_2d(states))[0]
            return np.argmax(self.predict(np.atleast_2d(states))[0]) # selects a greedy choice (max value computed by the network - exploitation)

    def add_experience(self, exp): # memorizes experience, if the max amount is exceeded the oldest element gets deleted
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet): #copies the weights of the dqn to the TrainNet
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(state, environment, TrainNet, TargetNet, epsilon, copy_step):
    environment.reset()
    rewards = 0
    iter = 0
    done = False
    observations = state
    losses = list()
    while not done: # observes until game is done 
        action = TrainNet.get_action(observations, epsilon) # TrainNet determines favorable action
        prev_observations = observations # saves observations
        result = environment.step(action)
        observations = result[0]
        reward = result[1]
        done = result[2]
        if result[3] == 1:
            won = True
        else:
            won = False
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
    return rewards, mean(losses), won #returns rewards and average

def main():
    environment = g.tictactoe()
    state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha = environment.variables
    # state: the initial state
    # gamma: discount factor, weights importance of future reward [0,1]
    # copy_step: the amount of episodes until the TargetNet gets updated
    # num_states: Amount of states, num_actions: Amount of actions
    # hidden_units: Amount of hidden neurons 
    # max_experiences: sets the maximum data stored as experience, if exceeded the oldest gets deleted
    # min_experiences: sets the start of the agent learning
    # batch_size: amount of data processed at once
    # alpha: learning rate, defines how drastically it changes weights
    
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
    N = 500000
    total_rewards = np.empty(N)
    epsilon = 0.99
    win_count = 0
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses, won = play_game(state, environment, TrainNet, TargetNet, epsilon, copy_step)
        if won:
            win_count += 1
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
            f = open("log.txt", "a")
            f.write((str(n)+";"+str(total_reward)+ ";"+str(epsilon)+";"+str(avg_rewards)+";"+ str(losses)+";"+ str(win_count))+"\n")
            f.close()
            win_count = 0
    print("avg reward for last 100 episodes:", avg_rewards)

    # Get current time and save models
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    checkpoint_path = "models/"+current_time+"-N."+str(N) # Model saved at "models/Y.m.d-H:M:S-N.amountOfEpisodes"
    # Save the models
    tf.saved_model.save(TrainNet.model, checkpoint_path+"/TrainNet")
    tf.saved_model.save(TargetNet.model, checkpoint_path+"/TargetNet")
    
    log.plot()

if __name__ == '__main__':
    f = open("log.txt", "w")
    f.close()
    main()



