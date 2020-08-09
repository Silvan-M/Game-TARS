import numpy as np
import tensorflow as tf
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


def play_game(environment, dqn):
    environment.reset()
    done = False
    observations = environment.state
    tie = False
    while not done:
        action = dqn.get_action(observations, 0) # Dqn determines favorable action
        result = environment.step_player(action)
        observations = result[0]
        reward = result[1]
        done = result[2]
        won = result[3]
        random_action = -1
        while reward == -0.1:
            random_action = random.randint(0,dqn.num_actions-1)
            result = environment.step_player(random_action)
            observations = result[0]
            reward = result[1]
            done = result[2]
            won = result[3]
        if random_action != -1:
            print("Chose random action: "+str(random_action))
        if reward == 0.5:
            tie = True
        print(observations)
        print("DONE: ", done,"WON: ", won,"TIE: ", tie,"REWARD: ", reward)
    return won, tie

def main():
    environment = g.tictactoe()
    state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha = environment.variables

    dqn = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
    
    model_name = "model.2020.08.09-17.57.56-I.100-N.1000"
    directory = "models/"+model_name+"/TrainNet/"
    tf.saved_model.load(directory)

    won, tie = play_game(environment, dqn)

    if tie:
        print("It's a tie!")
    elif won:
        print("You lost! The AI won!")
    else:
        print("You won!")

if __name__ == '__main__':
    while True:
        main()
        inp = input("Do you want to restart? (Y/n)")
        if inp != "Y" and inp != "y" and inp != "":
            break



