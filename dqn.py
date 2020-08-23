import numpy as np
import tensorflow as tf
import os
import datetime
from statistics import mean
import random
import log
import games as g

from tensorflow.python.framework import ops
ops.reset_default_graph()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, InputLayer
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time

class MyModel(tf.keras.Model): # class with format tensorflow.keras.model, has ability to group layers into an object with training and inference (interpreter) features.
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__() # adds model as subclass to add the ability to differ the behaviour in training and interference
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,)) # define inputlayers, shape = determines dimension of vector [tuple]
        self.hidden_layers = [] # clears the list of hidden layers
        for i in hidden_units: 
            self.hidden_layers.append(tf.keras.layers.Dense( 
                i, activation='relu', kernel_initializer='RandomNormal')) 
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


DISCOUNT = 0.999
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.epsilon = 0.9
        self.MIN_EPSILON = 0.2
        self.EPSILON_DECAY = 0.99
        self.ep_rewards = []

        self.ACTION_SPACE_SIZE = 9
        # Custom tensorboard object

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(InputLayer(9,))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(27))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(27))

        model.add(Dense(9, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            
            return


        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            print("Index: ",index,";Curren state: ", current_state,";Action: ",action,";Reward: ", reward, ";new_current_state: ",new_current_state, ";Done: ",done)
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        self.append_reward_and_decay(reward)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, (9)))[0] # CHANGE SHAPE TO OBJECT ORIENTED
        
    def get_action(self, state):
        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.get_qs(state))
        else:
            # Get random action
            action = np.random.randint(0, self.ACTION_SPACE_SIZE)
        return action

    def append_reward_and_decay(self, episode_reward):
         # Append episode reward to a list and log stats (every given number of episodes)
        self.ep_rewards.append(episode_reward)

        # Decay epsilon
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(9, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a


class exp_replay():
    def __init__(self, buffer_size= 1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *(9,)), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *(9,)), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class agent():
    def __init__(self, gamma=0.99, replace=100, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = exp_replay()
        self.batch_size = 64
        self.model = DDDQN()
        self.target_net = DDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(9)])

        else:
            actions = self.model.advantage(np.array([state]))
            action = np.argmax(actions)
            return action


    
    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)


    def update_target(self):
        self.target_net.set_weights(self.model.get_weights())     

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

        
    def train(self):
        if self.memory.pointer < self.batch_size:
            return 
        
        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        target = self.model.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.model.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
        self.model.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self):
        self.model.save("model.h5")
        self.target_net.save("target_model.h5")


    def load_model(self):
        self.model = load_model("model.h5")
        self.target_net = load_model("model.h5")
