# This is a commented DQN following the tutorial "How to Code Deep Q Learning in Tensorflow (Tutorial)" by "Machine Learning with Phil" (check it out on YouTube)

import os
import numpy as np
import tensorflow as tf

# The actual neuronal network
class dqn(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256, input_dims=(210,160,4), chkpt_dir='tmp/dqn'):
        # Learning Rate
        self.lr = lr
        # Name of DQN
        self.name = name
        # Number of Actions
        self.n_actions = n_actions
        # MORE ON THAT LATER
        self.fcl_dims = fcl_dims
        # Input dimension
        self.input_dims = input_dims
        # Instatiating a TensorFlow Session
        self.sess = tf.Session()
        # Add everything to the graph
        self.build_network()
        # This initalizes the Neuronal Network, TensorFlow requires this
        self.sess.run(tf.global_variables_initializer())
        # This will help saving all the models
        self.saver = tf.train.Saver()
        # Path to the saved 'checkpoints'
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')
        # Telling TensorFlow to keep track of all trainable variables 
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    
    # Build the network
    def build_net(self):
        # Encase everything in the scope of the networks name
        with tf.variable_scope(self.name):
            # PLACEHOLDER VARIABLES (needed to tell the DQN what inputs we have)
            # we input: stack of images from the game, actions that the agent took and the target value of the DQN
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs') 
            self.actions = tf.placeholder(tf.float32, shape=[None, *self.n_actions], name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None, *self.n_actions])
            # Naming placeholders and layers is important for debugging
            # Inputting "None" as shape will allow input to batches of stacked frames

            # BUILD THE LAYERS
            # Build the first layer and use varaiance_scaling_initializer(scale=2) like the deep mind team used
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32, strides=4, name='conv1', kernel_initalizer=tf.varaiance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)
            
            # Build the second layer and take the first as it's input
            conv2 = tf.layers.conv2d(input=conf1_activted, filters=64, kernel_size=(4,4), strides=2, name='conv2', kernel_initalizer=tf.varaiance_scaling_initializer(scale=2))
            conv2_activated = tf.nn.relu(conv2)

            # Build the third layer
            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters = 128, kernel_size=(3,3), strides=1, name='conv3', kernel_initalizer=tf.varaiance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)
            
            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims, activation=tf.nn.relu, kernel_initalizer=tf.varaiance_scaling_initializer(scale=2))

            # Get Q-values (state-action-pairs)
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions, kernel_initalizer=tf.varaiance_scaling_initializer(scale=2))
            # linear values (actual value) of Q
            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))
            # Calcualte Loss (q - q_target)^2
            # q_target: The optimal action it could have taken after a step
            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))
            # Training operation, form of gradient descent adam optimizer
            self.train_op = tf.train.AdamOptizmizer(self.lr).minimize(self.loss) # minimize the loss funcion
    
    # Training takes a very long time, therefore it's a great idea to have checkpoints, so you can stop at a certain time
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        # Look in the checkpoint file and load the graph into the graph of the current session
        self.saver.restore(self.sess, self.checkpoint_file)
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # Takes current session and outputs it to the current session 
        self.saver.save(self.sess, self.checkpoint_file)

    # The agent includes everything else, learning, memories, ...
    class Agent(object):
        # alpha = learning rate (between [0,1])
        # gamma = discount factor, weights importance of future reward (also between [0,1])
        # epsilon = how long it takes to explore (to take random action), decreased over time
        # mem_size = how many transitions to store in memory

        def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size, replace_target=5000, input_dims=(210, 160, 4), q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
            # Creates list of numbers ranging from 0 to n_actions (n_actions-1 being the last element)
            self.action_space = [ i for i in range(self.n_actions)]

            # initialize all values
            self.n_actions = n_actions
            self.gamma = gamma
            self.mem_size = mem_size
            self.epsilon = epsilon
            self.batch_size = batch_size
            # A seperate neuronal network with fixed weights that will be updated in every x iteration to adjust q-target
            self.replace_target = replace_target

            # Counter that keeps track of the number of memories that are stored
            self.mem_cntr

            # Tell the neuronal network the value of the next action
            self.q_next = dqn(alpha, n_actions, input_dims, name='q_next', chkpt_dir=q_next_dir)
            self.q_eval = dqn(alpha, n_actions, input_dims, name='q_eval', chkpt_dir=q_eval_dir)
            
            # MEMORY: It saves state-action-rewards, new-states-transitions and the terminal-flags (tell if game is done)
            # Save a set of 4 stacked frames (transitions) by number memories
            self.state_memory = np.zeros((self.mem_size, *input_dims))
            self.new_state_memory = np.zero((self.mem_size, *input_dims))
            
            # Store the one hot encoding of the actions (one hot encoding: store categorical values binary matrices instead of as integers, otherwise the DQN would believe higher number equals 'better' category)
            # dtype=np.int8: int8 consumes less RAM,  save RAM space measure
            self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
            self.reward_memory = np.zeros(self.mem_size) # One-dimensional
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

        def store_transition(self, state, action, reward, state_, terminal):
            # index: Fill up the memory until fixed memory size of agent is filled up, when exceeding it should go back to the beginning and start overwriting it
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state

            # One hot encoding of action
            actions = np.zeros(self.n_actions)
            actions[action] = 1.0
            self.action_memory[index] = actions

            # Save other variables
            self.reward_memory[index] = reward
            self.new_state_memory[index] = state_
            self.new_state_memory[index] = terminal

            # Increase Memory Counter
            self.mem_cntr += 1

        def choose_action(self, state):
            rand = np.random.random()
            # If ramdom is smaller than epsilon, epsilon decreases over time, make a random action > exploration
            if rand < self.epsilon:
                # Take random action
                action = np.random.choice(self.action_space)
            else:
                # Take a action with maximum output (greedy action)
                actions = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: state})
                actions = np.argmax(actions)
            return action


    def learn(self):
        # 
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        # Find out where memory ends
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, self.batch_size)
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        # Convert one hot ecoding back to integer
        action_values = np.array([0, 1 , 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: state_batch})
        q_next = self.q_next.sess.run(self.q_next.Q_values, feed_dict={self.q_next.input: new_state_batch})

        q_target = q_eval.copy()
        q_target[:, action_indices] = reward_batch + self.gamma*np.max(q_next, axis=1)*terminal_batch

        _ = self.q_eval.sess.run(self.q_eval.train_op, feed_dict={self.q_eval.input: state_batch, self.q_eval.actions: action_batch, self.q_eval.q_target: q_target})

        if self.mem_cntr > 100000:
            if self.epsilon > 0.01:
                self.epsilon *= 0.99999999
            elif self.epsilon <= 0.01:
                self.epsilon = 0.01
        
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t,e))




            