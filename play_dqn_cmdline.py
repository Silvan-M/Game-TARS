import numpy as np
import tensorflow as tf
import os
import datetime
from statistics import mean
import random
import log
import games as g
import dqn as dqn

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
    state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = environment.variables

    dqn = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
    
    model_name = "model.2020.08.12-19.42.21-I.100-N.1000"
    directory = "tictactoe/models/"+model_name+"/TrainNet/"
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



