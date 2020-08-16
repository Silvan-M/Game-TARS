import random
import time

class tictactoe:
    def __init__(self):
        self.illegalcount = 0

        # Variables
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state = state
        gamma = 0.9
        copy_step = 50
        num_state = 9
        num_actions = 9
        hidden_units = [27,27,27]
        max_experience = 50000
        min_experience = 100
        batch_size = 1
        alpha = 0.01
        epsilon = 0.9
        min_epsilon = 0.2
        decay = 0.99
        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha, epsilon, min_epsilon, decay]

        # Enable debugging if necessary
        self.debugging = False
        
        # TicTacToe specific variables
        self.reward_tie = 0.5
        self.reward_win = 1
        self.reward_lose = -1
        self.reward_illegal_move = -10
        self.reward_legal_move = 0
    
    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def checkWhoWon(self):
        #check if previous move caused a win on vertical line 
        for i in range(0,3):
            v = i*3
            if self.state[v] == self.state[1+v] == self.state[2+v] != 0:
                return True, self.state[v]

        #check if previous move caused a win on horizontal line 
        for i in range(0,3):
            if self.state[i] == self.state[3+i] == self.state[6+i] != 0:
                return True, self.state[i]

        #check if previous move was on the main diagonal and caused a win
        if self.state[0] == self.state[4] == self.state[8] != 0:
            return True, self.state[0]

        #check if previous move was on the secondary diagonal and caused a win
        if self.state[2] == self.state[4] == self.state[6] != 0:
            return True, self.state[2]

        return False, 0 
    
    def step_player(self, action)  -> list:
        reward = 0
        won = False
        illegalmove = False

        if self.state[action] != 0:
            reward = self.reward_illegal_move
            illegalmove = True
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = self.reward_legal_move
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        while (0 in self.state) and not illegalmove and not done:
            print(self.state[0:3], "   ", [0,1,2])
            print(self.state[3:6], "   ", [3,4,5])
            print(self.state[6:9], "   ", [6,7,8])
            var = int(input("Choose which field to set, starting on the top left with 0.\nI choose: ")) # 0 = empty, 1 = AI, 2 = player
            if self.state[var] == 0:
                self.state[var] = 2
                break
            else: 
                print("Please make a valid move.")

        # Check again
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = self.reward_win
                won = True
            else:
                reward = self.reward_lose
        # Tie
        if 0 not in self.state:
            done = True
            reward = self.reward_tie
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        return [self.state, reward, done, won]  

    def step_random(self, action)  -> list:
        reward = 0
        won = False
        lose = False
        illegalmove = False
        
        if self.state[action] != 0:
            reward = self.reward_illegal_move
            illegalmove = True
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = self.reward_legal_move
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        while (0 in self.state) and not illegalmove and not done:
            var = random.randint(0,8) # 0 = empty, 1 = AI, 2 = player
            if self.state[var] == 0:
                self.state[var] = 2
                break
        
        # Check again
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = self.reward_win
                won = True
                lose = False
            else:
                lose = True
                won = False
                reward = self.reward_lose

        # Tie
        if (0 not in self.state) and not done:
            done = True
            reward = self.reward_tie
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        if done and self.debugging:
            print(self.state[0:3], "   ", [0,1,2])
            print(self.state[3:6], "   ", [3,4,5])
            print(self.state[6:9], "   ", [6,7,8])
            print("Winner: ", winner,"Reward: ", reward,"Won: ", won,"Lose: ", lose)
        return [self.state, reward, done, won, lose, illegalmove]
    
    def step_once(self, action, activePlayer)  -> list:
        reward = 0
        won = False
        lose = False
        illegalmove = False
        activeTicTacToePlayer = activePlayer + 1
        
        if self.state[action] != 0:
            reward = self.reward_illegal_move
            illegalmove = True
            self.illegalcount +=1
        else:
            self.state[action] = activeTicTacToePlayer
            reward = self.reward_legal_move
            if activePlayer == 0:
                activePlayer = 1
            else:
                activePlayer = 0
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = self.reward_win
                won = True
                lose = False
            else:
                lose = True
                won = False
                reward = self.reward_lose

        # Tie
        if (0 not in self.state) and not done:
            done = True
            reward = self.reward_tie
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        if done and self.debugging:
            print(self.state[0:3], "   ", [0,1,2])
            print(self.state[3:6], "   ", [3,4,5])
            print(self.state[6:9], "   ", [6,7,8])
            print("Done: ", done,"Winner: ", winner,"Reward: ", reward,"Won: ", won,"Lose: ", lose)
        return [self.state, reward, done, won, lose, illegalmove, activePlayer]

class ultimate_tictactoe:
    def __init__(self):
        
        # Variables
        state = [[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.state = state
        gamma = 0.99
        copy_step = 25
        num_state = 9
        num_actions = 9
        hidden_units = [200, 200]
        max_experience = 10000
        min_experience = 100
        batch_size = 32
        alpha = 1e-2
        epsilon = 0.9
        min_epsilon = 0.1
        decay = 0.99
        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha, epsilon, min_epsilon, decay]
    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]