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
import glob
import os
import games as g
import dqn as dqn
import min_max_alg as mma

def snakeGetSeed(start, stop):
        snake = g.snake()
        score = 0

        # Initialize DQN
        state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = snake.variables
        originalState = state

        snakeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)

        directory = "snake/models/pretrained_model/TrainNet/"
        snakeDQN.model = tf.saved_model.load(directory)
        highscore = 0

        for seed in range(start, stop):
            done = False
            score = 0
            state = originalState
            snake.reset()
            random.seed(seed)
            while not done:
                action = snakeDQN.get_action(np.array(state),0)
                done, reward, state =  snake.step(action)
                field = snake.field

                if reward == snake.reward_apple:
                    score += 1

                if done:
                    if score > highscore:
                        print("Seed: {:5}, Score: {:5}".format(seed, score))
                        highscore = score


def spaceInvaderGetSeed(start, stop):
    spaceInvader = g.space_invader()
    highscore = 0

    # Initialize DQN
    state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = spaceInvader.variables
    state = [0]*num_states
    prevState = [0]*num_states

    spaceinvaderDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
                
    directory = "spaceinvader/models/pretrained_model/TrainNet/"
    spaceinvaderDQN.model = tf.saved_model.load(directory)
    for seed in range(start, stop):
        random.seed(seed)
        spaceInvader = g.space_invader()
        action = ['N', False]
        shoot_buffer = 0
        # internal score
        score = spaceInvader.score
        health = spaceInvader.health
        while True:
            health = spaceInvader.health
            # check if health is zero and if so go to next screen
            if health <= 0:
                if score[3] > highscore:
                        print("Seed: {:5}, Score: {:5}".format(seed, score[3]))
                        highscore = score[3]
                break
            inp = state

            if spaceinvaderDQN.batch_size > 1:
                # Simulate batch size of 2
                inp = [np.asarray(prevState).flatten(), np.asarray(state).flatten()]

            action = spaceinvaderDQN.get_action(np.array(inp), 0) # TrainNet determines favorable action

            shoot_buffer += 1
            convAction = ['N', False]
            if action == 0:
                convAction = ['L', False]
            elif action == 1:
                convAction = ['R', False]
            elif action == 2:
                if shoot_buffer >= 5:
                    convAction = ['N', True]
                    shoot_buffer = 0
            elif action == 3:
                convAction = ['N', False]

            prevState = state
            _, state = spaceInvader.step(convAction)
            

print("Finding snake highscores:")
snakeGetSeed(0,1000)
print("Finding spaceinvader highscores:")
spaceInvaderGetSeed(0,100)