import random
class tictactoe:
    def __init__(self):
        self.illegalcount = 0
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.variables = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 0.99, 25, 9, 9, [200, 200], 10000, 100, 32, 1e-2]
        # Input: [state, gamma, copy_step, num_states, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha]
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
         
    def step(self, action)  -> list:
        reward = 0

        if self.state[action] != 0:
            reward = -50
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = 50
        
        while True and (0 in self.state):
            var = random.randint(0,8) # 0 = empty, 1 = AI, 2 = player
            if self.state[var] == 0:
                self.state[var] = 2
                break
        
        # if game is done, end the game
        done, winner = self.checkWhoWon()

        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                reward = 10
            else:
                reward = -10
        # Tie
        if 0 not in self.state:
            done = True
            reward = +10
        
        # print("Done: "+str(done)+", Winner: "+str(winner), "Reward: "+str(reward))
        # print(self.state)
        return [self.state, reward, done]
    
    def state(self):
        return(self.state)

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
