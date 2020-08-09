import random

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
        hidden_units = []
        max_experience = 10000
        min_experience = 100
        batch_size = 128
        alpha = 0.1
        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha]
    
    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def checkWhoWon(self):
        #check if previous move caused a win on vertical line 
        for i in range(0,2):
            v = i*3
            if self.state[v] == self.state[1+v] == self.state[2+v] != 0:
                return True, self.state[v]

        #check if previous move caused a win on horizontal line 
        for i in range(0,2):
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

        if self.state[action] != 0:
            reward = 0
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = 0
        
        while True and (0 in self.state):
            var = int(input()) # 0 = empty, 1 = AI, 2 = player
            if self.state[var] == 0:
                self.state[var] = 2
                break
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = 500
                won = True
            else:
                reward = -500
        # Tie
        if 0 not in self.state:
            done = True
            reward = 0
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        return [self.state, reward, done, won]  

    def step(self, action)  -> list:
        reward = 0
        won = False
        lose = False
        illegalmove = False
        
        if self.state[action] != 0:
            reward = 0
            illegalmove = True
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = 0
        
        while True and (0 in self.state) and not illegalmove:
            var = random.randint(0,8) # 0 = empty, 1 = AI, 2 = player
            if self.state[var] == 0:
                self.state[var] = 2
                break
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = 1
                won = True
                lose = False
            else:
                lose = True
                won = False
                reward = -1

        # Tie
        if (0 not in self.state) and not done:
            done = True
            reward = 0.5
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        debugging = False
        if done and debugging:
            print(self.state[0:3])
            print(self.state[3:6])
            print(self.state[6:9])
            print("Winner: ", winner,"Reward: ", reward,"Won: ", won,"Lose: ", lose)
        return [self.state, reward, done, won, lose]
    

# # Testing
# a = tictactoe()
# for i in range(0,15):
#     print(a.state[0:3])
#     print(a.state[3:6])
#     print(a.state[6:9])
#     inp = input("Select move: ")
#     state, reward, done  = tictactoe.step(a, int(inp))
#     if done:
#         print("State: "+str(state)+", Reward: "+str(reward)+", Done: "+str(done))
#         break

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
        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha]
    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]