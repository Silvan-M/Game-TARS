import random
import time
import min_max_alg as mma

class tictactoe:
    def __init__(self):
        self.illegalcount = 0

        # Variables
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state = state
        gamma = 0.1
        copy_step = 1
        num_state = 18
        num_actions = 9
        hidden_units = [27*9]
        max_experience = 50000
        min_experience = 100
        batch_size = 1
        alpha = 0.01
        epsilon = 1
        min_epsilon = 0.01
        decay = 0.99
        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha, epsilon, min_epsilon, decay]

        # Enable debugging if necessary
        self.debugging = False
        
        # TicTacToe rewards
        self.reward_tie = 5000
        self.reward_win = 5000
        self.reward_lose = -600
        self.reward_illegal_move = 0
        self.reward_legal_move = 0
        self.reward_immediate_preset = 0
        self.reward_immediate_prevent = 0
    
    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def giveImmediateReward(self, beforeState, afterState):
        beforeStateManipulated = [beforeState[0:3],beforeState[3:6],beforeState[6:9]]
        afterStateManipulated = [afterState[0:3],afterState[3:6],afterState[6:9]]

        beforePreset = self.checkForTwoPreset(beforeStateManipulated)
        afterPreset = self.checkForTwoPreset(afterStateManipulated)

        beforePrevent = self.checkForTwoPrevent(beforeStateManipulated)
        afterPrevent = self.checkForTwoPrevent(afterStateManipulated)
        reward = 0
        if afterPreset-beforePreset > 0:
            reward += self.reward_immediate_preset
        if afterPrevent-beforePrevent > 0:
            reward += self.reward_immediate_prevent
        return reward

    def checkForTwoPrevent(self, sM):
        count = 0
        for i in range(3):
            # HORIZONTAL

            # Check horizontal 1 1 0
            if (sM[i][0] == sM[i][1] == 2) and sM[i][2] == 1:
                count += 1
            
            # Check horizontal 1 0 1
            if (sM[i][0] == sM[i][2] == 2) and sM[i][1] == 1:
                count += 1
            
            # Check horizontal 0 1 1
            if (sM[i][2] == sM[i][1] == 2) and sM[i][0] == 1:
                count += 1
            
            # VERTICAL

            # Check vertical 1 1 0
            if (sM[0][i] == sM[1][i] == 2) and sM[2][i] == 1:
                count += 1
            
            # Check vertical 1 0 1
            if (sM[0][i] == sM[2][i] == 2) and sM[1][i] == 1:
                count += 1
            
            # Check vertical 0 1 2
            if (sM[2][i] == sM[1][i] == 2) and sM[0][i] == 1:
                count += 1

        # DIAGONALS
        
        # Diagonal LT - RB
        # Check diagonal 1 1 0
        if (sM[0][0] == sM[1][1] == 2) and sM[2][2] == 1:
            count += 1

        # Check diagonal 1 0 1
        if (sM[0][0] == sM[2][2] == 2) and sM[1][1] == 1:
            count += 1

        # Check diagonal 0 1 1
        if (sM[1][1] == sM[2][2] == 2) and sM[0][0] == 1:
            count += 1

        # Diagonal RT - LB
        # Check diagonal 1 1 0
        if (sM[0][2] == sM[1][1] == 2) and sM[2][0] == 1:
            count += 1

        # Check diagonal 1 0 1
        if (sM[0][2] == sM[2][0] == 2) and sM[1][1] == 1:
            count += 1

        # Check diagonal 0 1 1
        if (sM[1][1] == sM[2][0] == 2) and sM[0][2] == 1:
            count += 1
        return count

        
    def checkForTwoPreset(self, sM):
        count = 0
        for i in range(3):
            # HORIZONTAL

            # Check horizontal 1 1 0
            if (sM[i][0] == sM[i][1] == 1) and sM[i][2] == 0:
                count += 1
            
            # Check horizontal 1 0 1
            if (sM[i][0] == sM[i][2] == 1) and sM[i][1] == 0:
                count += 1
            
            # Check horizontal 0 1 1
            if (sM[i][2] == sM[i][1] == 1) and sM[i][0] == 0:
                count += 1
            
            # VERTICAL

            # Check vertical 1 1 0
            if (sM[0][i] == sM[1][i] == 1) and sM[2][i] == 0:
                count += 1
            
            # Check vertical 1 0 1
            if (sM[0][i] == sM[2][i] == 1) and sM[1][i] == 0:
                count += 1
            
            # Check vertical 0 1 1
            if (sM[2][i] == sM[1][i] == 1) and sM[0][i] == 0:
                count += 1

        # DIAGONALS
        
        # Diagonal LT - RB
        # Check diagonal 1 1 0
        if (sM[0][0] == sM[1][1] == 1) and sM[2][2] == 0:
            count += 1

        # Check diagonal 1 0 1
        if (sM[0][0] == sM[2][2] == 1) and sM[1][1] == 0:
            count += 1

        # Check diagonal 0 1 1
        if (sM[1][1] == sM[2][2] == 1) and sM[0][0] == 0:
            count += 1

        # Diagonal RT - LB
        # Check diagonal 1 1 0
        if (sM[0][2] == sM[1][1] == 1) and sM[2][0] == 0:
            count += 1

        # Check diagonal 1 0 1
        if (sM[0][2] == sM[2][0] == 1) and sM[1][1] == 0:
            count += 1

        # Check diagonal 0 1 1
        if (sM[1][1] == sM[2][0] == 1) and sM[0][2] == 0:
            count += 1
        return count

    def convert0neHot(self, observation):
        oneHot = 18*[0]
        for i in range(9):
            if observation[i] == 1:
                oneHot[i] = 1
            else:
                oneHot[i+9] = 1
        return oneHot

    def isIllegalMove(self, action):
        return(self.state[action] != 0)

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


    def step(self, action, ran = False)  -> list:
        reward = 0
        won = False
        lose = False
        illegalmove = False
        before_state = self.state.copy()
        
        if self.state[action] != 0:
            reward = self.reward_illegal_move
            illegalmove = True
            self.illegalcount +=1
        else:
            self.state[action] = 1
            reward = self.reward_legal_move
        
        after_state = self.state.copy()

        # if game is done, end the game
        done, winner = self.checkWhoWon()
        
        while (0 in self.state) and not illegalmove and not done:
            if ran:
                var = random.randint(0,8) # 0 = empty, 1 = AI, 2 = player
            else:
                var = mma.GetMove(self.state, False)
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
        
        if not done and not illegalmove:
            reward = self.giveImmediateReward(before_state,after_state)
            # if reward != 0:
            #     print("BEFORE: ")
            #     print(before_state)
            #     print("AFTER: ")
            #     print(after_state)
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

class snake:
    def __init__(self):
        self.illegalcount = 0
        self.mode = 0 # Mode 0: 12 inputs, see below; Mode 1: input the complete field

        # Important field size variable
        self.field_size = 20 # 20x20 snake grid

        # Variables
        if self.mode == 0:
            self.state = [0]*12 # 0-4: Apple, 5-8: Obstacle, 9-12: Direction of Snake head | Index: 0=Above, 1=Right, 2=Below, 3=Left
            self.batch_size = 1
        elif self.mode == 1:
            self.state = [0]*(self.field_size**2)*2
            self.batch_size = 2
        gamma = 0.9
        copy_step = 50
        num_state = len(self.state)
        num_actions = 4 # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        hidden_units = [27*9]
        max_experience = 50000
        min_experience = 100
        alpha = 0.01
        epsilon = 1
        min_epsilon = 0.01
        decay = 0.99
        self.variables = [self.state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, self.batch_size, alpha, epsilon, min_epsilon, decay]

        # Enable debugging if necessary
        self.debugging = False
        
        # Snake variables
        self.apple = random.randint(0, self.field_size**2-1)
        self.snake = [110]
        self.prevAction = 2

        # Snake rewards
        self.reward_apple = 10 # Snake collects apple
        self.reward_closer = 1 # Snake gets closer to the apple
        self.reward_further = -2 # Snake gets further away from the apple
        self.reward_death = -100 # Snake dies (runs into wall or itself)
        self.reward_opposite_dir = -1 # Snake tries to go in opposite direction it's heading (not possible in snake)

        self.updateFieldVariable()

    def reset(self):
        self.field_size = 20 # 20x20 snake grid
        self.apple = random.randint(0, self.field_size**2-1)
        self.snake = [110]
        self.prevAction = 2
        self.updateFieldVariable()

    def step(self, action):
        # Evaluate action, detect if it hits the wall or itself
        index = self.getIndexOfAction(action)
        opposite = False

        # If snake hits wall or itself give negative reward
        if index == -1:
            # Check if snake wants to go in opposite direction it's heading
            if (self.prevAction == 0 and action == 2) or (self.prevAction == 2 and action == 0) or (self.prevAction == 1 and action == 3) or (self.prevAction == 3 and action == 1):
                reward = self.reward_opposite_dir
                opposite = True
                index = self.getIndexOfAction(self.prevAction)
                if index == -1:
                    return True, self.reward_death, self.getState(action)
            else:
                return True, self.reward_death, self.getState(action)

        # Initialize reward for later manipulation
        if not opposite:
            reward = 0

        # Snake has eaten an apple
        if index == self.apple:
            self.snake.append(index)
            while self.apple in self.snake:
                self.apple = random.randint(0, self.field_size**2-1)
            reward = self.reward_apple
        else:
            if self.gotCloserCheck(action):
                reward = self.reward_closer
            else:
                reward = self.reward_further
            self.snake.append(index)
            self.snake.pop(0)
        
        self.updateFieldVariable()
        
        if not opposite:
            self.prevAction = action
        return False, reward, self.getState(action)

    def updateFieldVariable(self):
        # Create empty field of field_size
        self.field = [0]*(self.field_size**2)

        # Create apple
        self.field[self.apple] = 2
        
        # Create snake
        for i in self.snake:
            self.field[i] = 1

    def gotCloserCheck(self, action):
        # Check where the apple currently is
        snakeHead = self.snake[-1]
        distanceTopWall = snakeHead//self.field_size
        distanceLeftWall = snakeHead-distanceTopWall*self.field_size
        if self.apple < distanceTopWall*self.field_size:
            # Apple is above
            if action == 0:
                return True
        elif self.apple > (distanceTopWall+1)*self.field_size:
            # Apple is below
            if action == 2:
                return True
        if ((self.apple+self.field_size) % self.field_size) > distanceLeftWall:
            # Apple is to the right
            if action == 1:
                return True
        elif ((self.apple+self.field_size) % self.field_size) < distanceLeftWall:
            # Apple is to the left
            if action == 3:
                return True
        return False


    def getIndexOfAction(self, action):
        snakeHead = self.snake[-1]

        # Moving up
        if action == 0:
            # If snake runs into wall return -1
            if snakeHead < self.field_size:
                return -1
            else:
                # Calculate new index, if the snake hits itself return -1
                index = snakeHead - self.field_size
                if index in self.snake:
                    return -1
                else:
                    return index
        # Moving right
        elif action == 1:
            # If snake runs into wall return -1
            if snakeHead in [i*self.field_size-1 for i in range(1,self.field_size+1)]:
                return -1
            else:
                # Calculate new index, if the snake hits itself return -1
                index = snakeHead + 1
                if index in self.snake:
                    return -1
                else:
                    return index
        # Moving down
        elif action == 2:
            # If snake runs into wall return -1
            if snakeHead >= self.field_size**2-self.field_size:
                return -1
            else:
                # Calculate new index, if the snake hits itself return -1
                index = snakeHead + self.field_size
                if index in self.snake:
                    return -1
                else:
                    return index
        # Moving left
        else:
            # If snake runs into wall return -1
            if (snakeHead in [i*self.field_size for i in range(0,self.field_size)]):
                return -1
            else:
                # Calculate new index, if the snake hits itself return -1
                index = snakeHead - 1
                if index in self.snake:
                    return -1
                else:
                    return index
    
    def getState(self, action):
        if self.mode == 0:
            # Check if the apple is above, right, below or to the left of the snakeHead
            apple = [0]*4
            snakeHead = self.snake[-1]
            distanceTopWall = snakeHead//self.field_size
            distanceLeftWall = snakeHead-distanceTopWall*self.field_size
            if self.apple < distanceTopWall*self.field_size:
                # Apple is above
                apple[0] = 1
            elif self.apple > (distanceTopWall+1)*self.field_size:
                # Apple is below
                apple[2] = 1
            
            if ((self.apple+self.field_size) % self.field_size) > distanceLeftWall:
                # Apple is to the right
                apple[1] = 1
            elif ((self.apple+self.field_size) % self.field_size) < distanceLeftWall:
                # Apple is to the left
                apple[3] = 1

            # Check if there is an obstacle directly above, right, below or to the left of the snakeHead
            obstacle = [0]*4
            for i in range(0,4):
                if self.getIndexOfAction(i) == -1:
                    obstacle[i] = 1

            # Determine the direction that the snake is currently heading
            direction = [0]*4
            direction[action] = 1

            return apple+obstacle+direction # 0-4: Apple, 5-8: Obstacle, 9-12: Direction of Snake head | Index: 0=Above, 1=Right, 2=Below, 3=Left
        elif self.mode == 1:
            self.updateFieldVariable()
            fieldSnake = self.field
            fieldApple = [0]*self.field_size**2
            fieldSnake[self.apple] == 0
            fieldApple[self.apple] == 1
            return fieldSnake+fieldApple
