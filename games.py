import random
import time
import min_max_alg as mma
import numpy as np
import floodfill as flood
import tensorflow as tf
class tictactoe:
    def __init__(self):
        # Ignore this (counters)
        self.illegalcount = 0
        self.fineTuneSelection = 4567
        self.fineTuneDetection = ""


        # Variables
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state = state
        gamma = 0.1
        copy_step = 1
        num_state = 18
        num_actions = 9
        hidden_units = [27*9*2]*2
        max_experience = 50000
        min_experience = 100
        batch_size = 1
        alpha = 0.01
        epsilon = 1
        min_epsilon = 0.01
        decay = 0.99
        
        # FINE TUNING: Here you can specify combinations which the AI cannot solve correctly, then the AI will be trained with those combinations
        # Notice: Empricial data has proven, that this method is usually leading to overshooting (network changes too drastically), so therefore don't expect much.
        self.fineTune = False
        self.combinationsToTest = ["576", "765", "675", "756", "0753", "7053"] # Insert combinations as strings eg. ["576","7053"]
        self.currentCombination = "576" # First combination to test (not so important, can be left empty)
        self.fineTuneAlpha = 0.0001 # Alpha should be lower when performing fine tune, so here you can specify alpha when fineTune is enabled
       
        if self.fineTune:
            alpha = self.fineTuneAlpha
            print("WARNING: RUNNING IN FINE TUNE MODE, ONLY CHECKING COMBINATIONS SPECIFIED")

        self.variables = [state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, batch_size, alpha, epsilon, min_epsilon, decay]

        # Same as fineTune, but automatically, the opponent remembers win strathegy
        self.detectFineTune = False

        if self.detectFineTune:
            if len(self.combinationsToTest) == 0:
                self.combinationsToTest = ["407"]
            self.fineTune = True

        # Enable debugging if necessary
        self.debugging = False
        
        # TicTacToe rewards
        self.reward_tie = 5000
        self.reward_win = 6000
        self.reward_lose = -5000
        self.reward_illegal_move = 0
        self.reward_legal_move = 0
        self.reward_immediate_preset = 0
        self.reward_immediate_prevent = 2000
    
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

        if not self.fineTune:
            while (0 in self.state) and not illegalmove and not done:
                if ran:
                    if self.state[4] == 0 and random.random()>0.3:
                        var = 4
                    else:
                        var = random.randint(0,8) # 0 = empty, 1 = AI, 2 = player
                else:
                    var = mma.GetMove(self.state, False)
                if self.state[var] == 0:
                    self.state[var] = 2
                    break
        else:
            # FINE TUNING: It checks combinations specified at initialisation
            if self.currentCombination != "":
                var = int(self.currentCombination[0])
                self.currentCombination = self.currentCombination[1:]
                while (0 in self.state) and not illegalmove and not done:
                    if self.state[var] == 0:
                        self.fineTuneDetection += str(var)
                        self.state[var] = 2
                        break
                    else:
                        var = random.randint(0,8)
            else:
                var = random.randint(0,8)
                while (0 in self.state) and not illegalmove and not done:
                    if self.state[var] == 0:
                        self.fineTuneDetection += str(var)
                        self.state[var] = 2
                        break
                    else:
                        var = random.randint(0,8)


        
        # Check again
        done, winner = self.checkWhoWon()
        if done:
            #print('illegal moves: ' +str(self.illegalcount)+', winner: '+str(winner))
            if winner == 1:
                self.fineTuneDetection = ""
                reward = self.reward_win
                won = True
                lose = False
            else:
                self.combinationsToTest.append(self.fineTuneDetection)
                self.fineTuneDetection = ""
                if len(self.combinationsToTest) > 7:
                    self.combinationsToTest.pop(0)
                lose = True
                won = False
                reward = self.reward_lose

            
            # Fine tuning reset
            if self.fineTune:
                if random.random() <= 0.3:
                    self.currentCombination = np.random.choice(self.combinationsToTest)

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
class space_invader:
    def __init__(self):
        self.illegalcount = 0
        self.prevPos = 0
        self.prevPosCounter = 0
        # Important field size variable
        # Variables
        self.lenState = 150
        self.height = 60
        self.state = np.zeros((self.lenState,self.height))
                                    #position[0], position[1] = position[1] , position[0]         
            # 1 = ship
            # 2 = enemy_lvl1    
            # 3 = enemy_lvl2   
            # 4 = enemy_lvl3  
            # 5 = ship_bullet
            # 6 = enemy_bullet 
            # 7 = enemy_leftovers (get replaced with air)
            # 8 = air (for bullets)
            # figures, c = 9 mark the center
        self.c = 9
        self.air = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # for clearing 
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

        self.ship = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                    [0,1,1,1,1,1,1,self.c,1,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]] 

        self.enemy_lvl1 = [[0,0,0,0,0,0,0,2,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,2,2,2,0,0,0,0,0,0],
                        [0,0,0,0,0,2,0,2,0,2,0,0,0,0,0],
                        [0,0,0,0,2,2,0,self.c,0,2,2,0,0,0,0],
                        [0,0,0,2,2,2,2,2,2,2,2,2,0,0,0],
                        [0,0,0,0,0,2,2,2,2,2,0,0,0,0,0],
                        [0,0,0,0,2,0,2,0,2,0,2,0,0,0,0]]

        self.enemy_lvl2 = np.array( [[0,0,0,0,0,0,3,3,3,0,0,0,0,0,0],
                        [0,0,0,0,0,3,3,3,3,3,0,0,0,0,0],
                        [0,0,0,0,3,3,3,3,3,3,3,0,0,0,0],
                        [0,0,0,3,0,3,3,self.c,3,3,0,3,0,0,0],
                        [0,0,0,3,0,3,3,0,3,3,0,3,0,0,0],
                        [0,0,0,0,3,3,3,3,3,3,3,0,0,0,0],
                        [0,0,0,0,0,3,0,3,0,3,0,0,0,0,0]])

        self.enemy_lvl3 = [[0,0,4,4,4,4,4,4,4,4,4,4,4,0,0],
                        [0,4,4,4,4,4,4,4,4,4,4,4,4,4,0],
                        [4,4,4,4,4,4,0,4,0,4,4,4,4,4,4],
                        [0,4,4,4,4,0,4,self.c,4,0,4,4,4,4,0],
                        [0,0,4,0,4,4,4,4,4,4,4,0,4,0,0],
                        [0,0,0,4,0,4,0,4,0,4,0,4,0,0,0],
                        [0,0,4,4,0,4,0,4,0,4,0,4,4,0,0]]

        self.enemy_leftovers = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,7,0,0,7,0,0,7,0,0,0,0],
                            [0,0,7,0,0,0,0,0,0,0,0,7,0,0,0],
                            [0,0,0,0,0,0,0,self.c,7,0,7,0,0,0,0],
                            [0,7,0,7,0,0,7,7,7,0,0,0,7,0,0],
                            [0,0,0,0,0,0,0,0,0,0,7,0,0,0,0],
                            [0,0,0,7,0,7,0,7,0,0,7,0,0,0,0]]
        self.enemy_lvl2 = np.rot90(self.enemy_lvl2)
        self.enemy_lvl1 = np.rot90(self.enemy_lvl1)
        self.enemy_lvl3 = np.rot90(self.enemy_lvl3)
        self.ship = np.rot90(self.ship)
        self.air = np.rot90(self.air)
        #just for enemy movement
        self.count = [0,'r',0] # [overall count, movement direction, movement's direction count]

        self.enemy_leftovers = np.rot90(self.enemy_leftovers)
        # self.oneHotState = [0] * self.lenState * 5
        # safes action, action[0] = R or L, action[1] = fire True or False
        self.action = ['N',False] # R = move right, L = move Left, True/False = Fire
        self.health = 3
        self.figures = [] # list with all object in the game [object, x_center, y_center]
        self.batch_size = 1
        
        # first round
        self.figures_set([65,55], 1)
        amount = random.randint(2,6)
        for i in range(amount):
                lvl = self.enemy_create()
                x = random.randint(20,140)
                y = random.randint(20, 40)
                self.figures_set([x,y], lvl)
        self.figures_set([10,40], 4)
        self.figures_set([135,40], 4)
        gamma = 0.9
        copy_step = 50
        #num_state = len(self.state)
        num_state = 5
        num_actions = 4 # 0 = Left, 1 = Right, 2 = Fire
        hidden_units = [1028]
        max_experience = 50000
        min_experience = 100
        alpha = 0.01
        epsilon = 1
        min_epsilon = 0.01
        decay = 0.99
        self.variables = [self.state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, self.batch_size, alpha, epsilon, min_epsilon, decay]

        # Enable debugging if necessary 
        self.debugging = False
        self.check = 0
        self.check_enemy = []
        self.message = None

        # Space invaders specific rewards
        self.reward_enemy_lvl_destroyed = 1000 # Ship destroys enemy 
        self.reward_all_enemies_destroyed = 2500 # Ship destroys all enemies
        self.reward_ship_hit = -2000 # Ship loses one life
        self.reward_ship_destroyed = -3000 # Ship gets destroyed
        self.reward_time_up = -5000 # Ship dies because time is up and enemies are up close
        self.reward_ship_targeted = 250 # Ship shoots when below an enemy
        self.reward_nothing_targeted = -50 # Ship shoots into oblivion
        self.reward_no_move = -500 # Ship does not move after 500 moves
        self.reward_no_move_but_incoming = -1000 # Ship does not move if missile incoming
        self.reward_side_penalty = -500 # Penalty if ship stays on one side (should stop tactics of not doing anything)
        self.score = [0,0,0,0,0] #lvl1, lvl2, lvl3, score, wave
        self.safe = []
        # States:
            # 0 = air
            # 1 = ship
            # 2 = enemy_lvl1
            # 3 = enemy_lvl2
            # 4 = enemy_lvl3
            # 5 = ship_bullet
            # 6 = enemy_bullet

    # calculates if an enemy shoots or not depending on the level
    def enemy_action(self,lvl):
        if lvl*2 > random.randint(0,150): #Chance of firing increases per lvl 
            return(True) #fire
        else:
            return(False) #not fire

    # creates enemy , highter lvl more unlikely to create
    def enemy_create(self): #creates randomly an enemy with random lvl
        ran = random.randint(0,100)
        if ran < 10:
            return(4)
        elif ran < 50 and ran > 10:
            return(3)
        else:
            return(2)
        # keeps track of all destroyed enemies and scores
        # score[0] = enemy lvl1 killed , 
        # score[1] = enemy lvl2 killed , 
        # score[2] = enemy lvl3 killed ,
        # score[3] = total score       ,  
        # score[4] = wave completed    , 
    
    def scoreboard(self,mode, lvl = None):
        if mode == 'de':
            if self.debugging:
                print('lvl '+str(lvl)+' score added')
                print(lvl-2)
            self.score[lvl-2] +=1
            self.score[3] += self.reward_enemy_lvl_destroyed * lvl**2
        elif mode == 'wa':
            self.score[4] += 1
        else:
            return(self.score)

    def figures_set (self,position,figure):
        # checks if the potential figure can be placed in the foreseen region, checked in every direction, dimension are [15,7]
        # ifso returns True, else False
        try:
            if not (position[0] + 7 < self.lenState) and (position[0] -7) > 0 and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                return(False)
            if figure == 1 and (position[0] + 7 < self.lenState) and (position[0] -7) > 0 and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.ship
                return(True)
            elif figure == 2 and (position[0] + 7 < self.lenState) and (position[0] -7 > 0)  and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.enemy_lvl1
                return(True)
            elif figure == 3 and (position[0] + 7 < self.lenState) and (position[0] -7 > 0)  and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.enemy_lvl2
                return(True)
            elif figure == 4 and (position[0] + 7 < self.lenState) and (position[0] -7 > 0)  and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.enemy_lvl3
                return(True)
            elif figure == 7 and (position[0] + 7 < self.lenState) and (position[0] -7 > 0)  and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.enemy_leftovers
                return(True)
            elif figure == 0 and (position[0] + 7 < self.lenState) and (position[0] -7 > 0)  and (position[1] - 4 > 0) and (position[1] + 4 < self.height): 
                self.state[ position[0]-7:position[0]+8,position[1]-3: position[1]+ 4] = self.air
                return(True)
            #bullets
            elif figure == 5 and (position[0] + 1 < self.lenState) and (position[0] -1 > 0)  and (position[1] -1 > 0) and (position[1] +1 < self.height):  
                self.state[position[0]][position[1]] = self.c
                #self.state[position[0]-1][position[1]] = 5
                #self.state[position[0]-1][position[1]-1] = 5
                #self.state[position[0]-1][position[1]+1] = 5
                #self.state[position[0]+1][position[1]] = 5
                self.state[position[0]-1][position[1]+1] = 5
                self.state[position[0]+1][position[1]+1] = 5
                return(True)

            elif figure == 6 and (position[0] + 1 < self.lenState) and (position[0] -1 > 0)  and (position[1] -1 > 0) and (position[1] +1 < self.height):  
                self.state[position[0]][position[1]] = self.c
                self.state[position[0]+1][position[1]] = 6
                self.state[position[0]+1][position[1]-1] = 6
                self.state[position[0]+1][position[1]+1] = 6
                self.state[position[0]-1][position[1]] = 6
                self.state[position[0]-1][position[1]-1] = 6
                self.state[position[0]-1][position[1]+1] = 6
                return(True)
            elif figure == 8 and (position[0] + 1 < self.lenState) and (position[0] -1 > 0)  and (position[1] -1 > 0) and (position[1] +1 < self.height):  
                self.state[position[0]][position[1]] = 0
                self.state[position[0]+1][position[1]] = 0
                self.state[position[0]-1][position[1]] = 0
                self.state[position[0]-1][position[1]-1] = 0
                self.state[position[0]-1][position[1]+1] = 0
                self.state[position[0]+1][position[1]-1] = 0
                self.state[position[0]+1][position[1]+1] = 0
                return(True)
            else:
                return(False)
        except IndexError:
            print('index error')
            return(False)

        # resets self.state
   
    def reset(self):
        self.state = np.zeros((self.lenState,self.height))
    
    def print(self):
        for i in range(len(self.state)):
            print(self.state[i])
        # makes one step
        # checks every discrete space for movement (projectile, ship) and intersections 
    def replacer(self):
                # 0 = air           --> 0
                # 1 = ship          --> 1
                # 2 = enemy_lvl1    --> -1
                # 3 = enemy_lvl2    --> -1
                # 4 = enemy_lvl3    --> -1
                # 5 = ship_bullet   --> -1
                # 6 = enemy_bullet  --> 1
                # 9 = center        --> 1
        transition = [[0,0],[1,1],[2,-1],[3,-1],[4,-1],[5,1],[6,0],[9,0]]
        self.return_state=[]
        for i in range(len(transition)):
            old = transition[i][0]
            new = transition[i][1]
            for x in range(len(self.state)):
                holder = np.zeros(len((self.state[x])),dtype="float32")
                self.return_state.append(holder)

                for y in range(len(self.state[x])):
                    if self.state[x][y]== old:
                        self.return_state[x][y] = new
        return self.return_state

    def get4State(self):
        additionalReward = 0

        # Checking side of nearest ship
        nearestShipLeft = 0
        nearestShipRight = 0

        xPosShips = []
        xPosPlayer = self.ship_figures[0][0]
        for i in self.enemy_fig:
            xPosShips.append(i[0])
        
        lowest = 50000 # Some high number as default
        for i in xPosShips:
            if abs(i - xPosPlayer) < lowest:
                lowest = abs(i - xPosPlayer)
                if i - xPosPlayer < 0:
                    nearestShipLeft = 1
                    nearestShipRight = 0
                else:
                    nearestShipLeft = 0
                    nearestShipRight = 1
        
        # Check for ships above
        shipAbove = 0
        for i in range(len(self.enemy_fig)):
            if abs(xPosPlayer-self.enemy_fig[i][0]) < 4:
                shipAbove = 1
        
        # Reward agent if he shoots when below an enemy
        if (shipAbove == 1) and (self.action[1] == True):
            additionalReward += self.reward_ship_targeted
        elif (shipAbove == 0) and (self.action[1] == True):
            additionalReward += self.reward_nothing_targeted

        # Checks if an enemy projectile is above the ship
        enemyProjectileAbove = 0
        for i in range(len(self.proj_fig)):
            if (abs(xPosPlayer-self.proj_fig[i][0]) < 8) and (self.proj_fig[i][2] == 6):
                enemyProjectileAbove = 1

        return additionalReward, [nearestShipLeft, nearestShipRight, shipAbove, enemyProjectileAbove]

    def get5State(self):
        additionalReward = 0

        # Checking side of nearest ship
        nearestShipLeft = 0
        nearestShipRight = 0

        xPosShips = []
        xPosPlayer = self.ship_figures[0][0]
        for i in self.enemy_fig:
            xPosShips.append(i[0])
        
        lowest = 50000 # Some high number as default
        for i in xPosShips:
            if abs(i - xPosPlayer) < lowest:
                lowest = abs(i - xPosPlayer)
                if i - xPosPlayer < 0:
                    nearestShipLeft = 1
                    nearestShipRight = 0
                else:
                    nearestShipLeft = 0
                    nearestShipRight = 1
        
        # Check for ships above
        shipAbove = 0
        for i in range(len(self.enemy_fig)):
            if abs(xPosPlayer-self.enemy_fig[i][0]) < 4:
                shipAbove = 1
        
        # Reward agent if he shoots when below an enemy
        if (shipAbove == 1) and (self.action[1] == True):
            additionalReward += self.reward_ship_targeted
        elif (shipAbove == 0) and (self.action[1] == True):
            additionalReward += self.reward_nothing_targeted

        # Checks if an enemy projectile is above the ship
        enemyProjectileLeft = 0
        enemyProjectileRight = 0
        lowestL = 50000 # Some high number as default
        lowestR = 50000 # Some high number as default
        for i in range(len(self.proj_fig)):
            if (xPosPlayer-self.proj_fig[i][0] < 0) and (self.proj_fig[i][2] == 6):
                if lowestR > self.proj_fig[i][0]-xPosPlayer:
                    lowestR = self.proj_fig[i][0]-xPosPlayer
            elif (xPosPlayer-self.proj_fig[i][0] > 0) and (self.proj_fig[i][2] == 6):
                if lowestL > xPosPlayer-self.proj_fig[i][0]:
                    lowestL = xPosPlayer-self.proj_fig[i][0]
            else:
                lowestR = 0

        if lowestR != 0:
            enemyProjectileRight += 1/(lowestR**2)*64
            enemyProjectileLeft += 1/(lowestL**2)*64
        else:
            enemyProjectileLeft = 128
            enemyProjectileRight = 128

        # Checks if an enemy projectile is above the ship, if it stays still punish agent
        enemyProjectileAbove = False
        for i in range(len(self.proj_fig)):
            if (abs(xPosPlayer-self.proj_fig[i][0]) < 8) and (self.proj_fig[i][2] == 6):
                enemyProjectileAbove = True
        if enemyProjectileAbove:
            if self.action[0] == 'N':
                additionalReward += self.reward_no_move_but_incoming

        return additionalReward, [nearestShipLeft, nearestShipRight, shipAbove, enemyProjectileLeft, enemyProjectileRight]

    def get5StatePos(self):
        additionalReward = 0

        # Checking side of nearest ship
        nearestShipLeft = 0
        nearestShipRight = 0

        xPosShips = []
        xPosPlayer = self.ship_figures[0][0]
        for i in self.enemy_fig:
            xPosShips.append(i[0])
        
        lowest = 50000 # Some high number as default
        for i in xPosShips:
            if abs(i - xPosPlayer) < lowest:
                lowest = abs(i - xPosPlayer)
                if i - xPosPlayer < 0:
                    nearestShipLeft = i
                else:
                    nearestShipRight = i
        
        # Check for ships above
        shipAbove = 0
        for i in range(len(self.enemy_fig)):
            if abs(xPosPlayer-self.enemy_fig[i][0]) < 4:
                shipAbove = 1
        
        # Reward agent if he shoots when below an enemy
        if (shipAbove == 1) and (self.action[1] == True):
            additionalReward += self.reward_ship_targeted
        elif (shipAbove == 0) and (self.action[1] == True):
            additionalReward += self.reward_nothing_targeted

        # Checks if an enemy projectile is above the ship
        enemyProjectileLeft = 0
        enemyProjectileRight = 0
        lowestL = 50000 # Some high number as default
        lowestR = 50000 # Some high number as default
        for i in range(len(self.proj_fig)):
            if (xPosPlayer-self.proj_fig[i][0] < 0) and (self.proj_fig[i][2] == 6):
                if lowestR > self.proj_fig[i][0]-xPosPlayer:
                    lowestR = self.proj_fig[i][0]-xPosPlayer
            elif (xPosPlayer-self.proj_fig[i][0] > 0) and (self.proj_fig[i][2] == 6):
                if lowestL > xPosPlayer-self.proj_fig[i][0]:
                    lowestL = xPosPlayer-self.proj_fig[i][0]
            else:
                lowestR = 0

        if lowestR != 0:
            enemyProjectileRight += 1/(lowestR**2)*64
            enemyProjectileLeft += 1/(lowestL**2)*64
        else:
            enemyProjectileLeft = 128
            enemyProjectileRight = 128

        return additionalReward, [xPosPlayer, nearestShipLeft, nearestShipRight, enemyProjectileLeft, enemyProjectileRight]

    def step(self, action):
        
        reward = 0
        self.action = action
        # splitting figures into 3 categories to avoid unnecessary loops
        self.ship_figures = [] # ship, enemies, projectiles

        self.enemy_fig = []
        self.proj_fig = []
        # iterating over x and y values
        for x_value in range(len(self.state)):
            for y_value in range(len(self.state[x_value])): # only applicable with object that move upwards or right to left
                # looking for centerpieces
                    if self.state[x_value][y_value]== 9:
                        if self.debugging == True:
                            print('Center found at: '+str(x_value)+','+str(y_value))
                            print('Marker: '+str(self.state[x_value+1][y_value+1]))
                        # associate object with type [ship, enemy_lvl1,enemy_lvl2, enemy_lvl3, enemy_bullet, ship_bullet, enemy_leftovers]
                        # saves the location and type in a list
                        if self.state[x_value+1][y_value+1] == 1:
                            self.ship_figures.append([x_value,y_value,self.state[x_value+1][y_value+1]])
                        elif self.state[x_value+1][y_value+1] == 5 or self.state[x_value+1][y_value+1] == 6:
                            self.proj_fig.append([x_value,y_value,self.state[x_value+1][y_value+1]])
                        else:
                            self.enemy_fig.append([x_value,y_value,self.state[x_value+1][y_value+1]])
                        if self.debugging:
                            print([['ship', self.ship_figures],['enemy', self.enemy_fig],['projectile', self.proj_fig]])
        self.safe = self.proj_fig 
        if self.debugging:
            if self.check != self.ship_figures:
                print('ship movement ',self.ship_figures)
                self.check = self.ship_figures
        # self.state gets reset, information stored in the figures list
        self.reset()
        # looks for movement 
        if len(self.ship_figures) != 0:
            if self.ship_figures[0][2] == 1: # ship
                #self.figures_set([figures[i][0],figures[i][1]],1)
                if action[0] == 'L': # if left key is pressed
                    # checks if its possible to move the ship in this direction
                    if self.figures_set([self.ship_figures[0][0] - 2 ,self.ship_figures[0][1]],1) == False:
                        self.figures_set([self.ship_figures[0][0] ,self.ship_figures[0][1]],1)
                    else: # if its possible, move
                        self.figures_set([self.ship_figures[0][0] - 2 ,self.ship_figures[0][1]],1)
                    if self.debugging:
                        print('Ship moved left')
                elif action[0] == 'R': # if right key is pressed
                    # checks if its possible to move the ship in this direction
                    if self.figures_set([self.ship_figures[0][0] + 2 ,self.ship_figures[0][1]],1) == False:
                        self.figures_set([self.ship_figures[0][0] ,self.ship_figures[0][1]],1)
                    else: # if its possible, move
                        self.figures_set([self.ship_figures[0][0]+2 ,self.ship_figures[0][1]],1)
                    if self.debugging:
                        print('Ship moved right')
                else: # if no movement key is pressed , no movement, only situation where ship can fire
                    self.figures_set([self.ship_figures[0][0],self.ship_figures[0][1]],1)
                if action[1] == True: # if up key is pressed, fire
                    self.figures_set([self.ship_figures[0][0],self.ship_figures[0][1]-5],5)
        # describes enemy behaviour, uses enemy_action to determine if they fire or not
        self.count[0] +=1
        speed = 150-self.score[4]
        if speed < 10:
            speed = 10
        if self.count[0] > speed:
            self.count[0] = 0  
            if self.debugging:
                print('Enemy moved down')
            for i in range(len(self.enemy_fig)):
                if self.enemy_fig[i][2] == 2: # enemy lvl 1
                    self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+1],2)
                    if self.enemy_action(1):
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                elif self.enemy_fig[i][2] == 3: # enemy lvl 2
                    self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+1],3)
                    if self.enemy_action(2):
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                elif self.enemy_fig[i][2] == 4: # enemy lvl 3
                    self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+1],4)
                    if self.enemy_action(3):
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)    
        elif self.count[1] == 'r':
            if self.count[2] > 50:
                self.count[1] = 'l'
                self.count[2] = 0
            else:
                self.count[2] += 1
            if self.count[2]%3487349789343 == 0:
                for i in range(len(self.enemy_fig)):
                    if self.enemy_fig[i][2] == 2: # enemy lvl 1
                        self.figures_set([self.enemy_fig[i][0]+1,self.enemy_fig[i][1]],2)
                        if self.enemy_action(1):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 3: # enemy lvl 2
                        self.figures_set([self.enemy_fig[i][0]+1,self.enemy_fig[i][1]],3)
                        if self.enemy_action(2):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 4: # enemy lvl 3
                        self.figures_set([self.enemy_fig[i][0]+1,self.enemy_fig[i][1]],4)
                        if self.enemy_action(3):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6) 
            else:
                for i in range(len(self.enemy_fig)):
                    if self.enemy_fig[i][2] == 2: # enemy lvl 1
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],2)
                        if self.enemy_action(1):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 3: # enemy lvl 2
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],3)
                        if self.enemy_action(2):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 4: # enemy lvl 3
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],4)
                        if self.enemy_action(3):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6) 
        elif self.count[1] == 'l':
            if self.count[2] > 50:
                self.count[1] = 'r'
                self.count[2] = 0
            else:
                self.count[2] += 1
                self.count[1] == 'r'
            if self.count[2]%3487349789343 == 0:
                for i in range(len(self.enemy_fig)):
                    if self.enemy_fig[i][2] == 2: # enemy lvl 1
                        self.figures_set([self.enemy_fig[i][0]-1,self.enemy_fig[i][1]],2)
                        if self.enemy_action(1):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 3: # enemy lvl 2
                        self.figures_set([self.enemy_fig[i][0]-1,self.enemy_fig[i][1]],3)
                        if self.enemy_action(2):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 4: # enemy lvl 3
                        self.figures_set([self.enemy_fig[i][0]-1,self.enemy_fig[i][1]],4)
                        if self.enemy_action(3):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6) 
            else:
                for i in range(len(self.enemy_fig)):
                    if self.enemy_fig[i][2] == 2: # enemy lvl 1
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],2)
                        if self.enemy_action(1):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 3: # enemy lvl 2
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],3)
                        if self.enemy_action(2):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6)
                    elif self.enemy_fig[i][2] == 4: # enemy lvl 3
                        self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]],4)
                        if self.enemy_action(3):
                            self.figures_set([self.enemy_fig[i][0],self.enemy_fig[i][1]+6],6) 
        else:
            print('none')  
        # describes projectile movement
        for i in range(len(self.proj_fig)): # checks all projectiles
            if self.proj_fig[i][2] == 5: # if ships projectile
                for y in range(len(self.enemy_fig)): # looks if it intersects with one of the enemies
                    if self.debugging:
                        print(str(y+1)+'. enemy checked from'+str(self.enemy_fig))
                        print(str(i+1)+'. projectile checked from'+str(self.proj_fig))
                        # checks intersection
                    if abs(self.proj_fig[i][0]-self.enemy_fig[y][0]) < 4 and abs(self.proj_fig[i][1]-self.enemy_fig[y][1]) < 4:
                        reward += self.reward_enemy_lvl_destroyed
                        if self.message != self.enemy_fig[y]:
                            self.message = self.enemy_fig[y]
                        
                        if self.debugging:
                            print('Enemy ship destroyed at',self.enemy_fig[y])
                            print(self.enemy_fig[y])
                        # enemy deleted
                        self.state[self.enemy_fig[y][0]][self.enemy_fig[y][1]] = 0
                        if self.enemy_fig[y][2] != 9: # checks if no centerpiece
                            # adds according score to the scoreboard
                            self.scoreboard( 'de', int(self.enemy_fig[y][2]))
                            #self.enemy_fig.pop(y)
                    elif self.proj_fig[i][2] == 5: # ships projectile
                        # if no intercept move normally
                        self.figures_set([self.proj_fig[i][0],self.proj_fig[i][1]-1],self.proj_fig[i][2])
            if self.proj_fig[i][2] == 6: # checks if its an enemy projectile
                for y in range(len(self.ship_figures)):
                    # looks for intersection with ship
                    if abs(self.proj_fig[i][0]-self.ship_figures[y][0]) < 8 and abs(self.proj_fig[i][1]-self.ship_figures[y][1])< 4:
                        if self.debugging:
                            print('Ship destroyed ')
                        reward += self.reward_ship_hit
                        # if intersected lose one healthpoint
                        self.health -= 1
                    elif self.proj_fig[i][2] == 6: # enemys projectile
                        # if no intercept move normally 
                        self.figures_set([self.proj_fig[i][0],self.proj_fig[i][1]+1],self.proj_fig[i][2])
            # looks if enemy ship is at the bottom
            for i in range(len(self.enemy_fig)):
                if self.enemy_fig[i][1] > 50:
                    self.health = 0
            # if no enemies are on the field
        
        if len(self.enemy_fig) == 0:
            reward += self.reward_all_enemies_destroyed
            # add points for completing the wave
            self.scoreboard('wa') 
            # makes list to check
            create = []
            # make random lvl
            lvl = self.enemy_create()
            # make random position
            x = random.randint(20,140)
            y = random.randint(20, 40)
            # create
            self.figures_set([x,y], lvl)
            create.append([x,y])
            # make random amount of enemies
            amount = random.randint(0,30)
            for i in range(amount):
                do = True
                # makes random enemy with random position
                lvl = self.enemy_create()
                x = random.randint(20,140)
                y = random.randint(20, 40)
                # checks if generated enemy does not intersect with any other enemy previously created
                for c in range(len(create)):
                    if abs(x - create[c][0]) > 16 or abs(y - create[c][1]) > 16:
                        do = True
                    else:
                        do = False
                        break
                if do:
                    # if no intersection, create
                    lvl = self.enemy_create()
                    self.figures_set([x,y], lvl )
                    create.append([x,y])
            create = []      

        '''if len(self.enemy_fig) != len(self.check_enemy):
            self.check_enemy == self.enemy_fig
            print('enemies: ',str(self.enemy_fig))'''
        
        if len(self.ship_figures) != 0:
            additionalReward, returnState = self.get5State()

            if self.prevPos == self.ship_figures[0]:
                self.prevPosCounter += 1
            else:
                self.prevPosCounter = 0

            if self.prevPosCounter >= 500:
                # Agent did not move, let's punish him
                reward += self.reward_no_move
            
            self.prevPos = self.ship_figures[0]

            # Side Reward
            if (self.ship_figures[0][0] == 9) or (self.ship_figures[0][0] == 141):
                reward += self.reward_side_penalty
        else:
            additionalReward, returnState = 0, [0]*self.variables[3]
            self.health = 0
            print("Ship figures empty!")
        
        reward += additionalReward

        self.ship_figures = []
        self.enemy_fig = []
        self.proj_fig = []

        return reward, returnState
        # return reward, self.replacer() #(tf.one_hot(self.replacer(), len(self.state),axis = 1))

class snake:
    def __init__(self):
        self.illegalcount = 0
        self.mode = 0 # Mode 0: 12 inputs, see below; Mode 1: input the complete field; Mode 2: Snake head centered

        # Important field size variable
        self.field_size = 20 # field_size x field_size snake grid

        # Variables
        if self.mode == 0:
            self.state = [0]*12 # 0-4: Apple, 5-8: Obstacle, 9-12: Direction of Snake head | Index: 0=Above, 1=Right, 2=Below, 3=Left
            self.batch_size = 1
        elif self.mode == 1:
            self.state = [0]*((self.field_size**2)*2)
            self.batch_size = 2
        elif self.mode == 2:
            outPutFieldSize = (self.field_size//2)*2-1
            self.state = [0]*((outPutFieldSize**2)*2)
            self.batch_size = 1

        gamma = 0.9
        copy_step = 50
        num_state = len(self.state)
        num_actions = 4 # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        hidden_units = [27*9]
        max_experience = 50000
        min_experience = 100
        alpha = 0.01
        epsilon = 1
        min_epsilon = 0.05
        decay = 0.999
        self.variables = [self.state, gamma, copy_step, num_state, num_actions, hidden_units, max_experience, min_experience, self.batch_size, alpha, epsilon, min_epsilon, decay]

        # Enable debugging if necessary
        self.debugging = False
        
        # Snake variables
        self.apple = random.randint(0, self.field_size**2-1)
        self.snake = [int(self.field_size**2/2)]
        self.prevAction = 2
        self.memory = []

        # Snake rewards
        self.reward_apple = 1000 # Snake collects apple
        self.reward_closer = 10 # Snake gets closer to the apple
        self.reward_further = -15 # Snake gets further away from the apple
        self.reward_death = -100 # Snake dies (runs into wall or itself)
        self.reward_opposite_dir = -15 # Snake tries to go in opposite direction it's heading (not possible in snake)
        self.reward_opposite_dir_topoff = self.field_size**2*0.40 # The score after which no opposite direction penalty will be given
        self.reward_repetitive = -50 # If the snake ends up in the exact same situation as in the last 6 steps
        self.reward_enclosing = -50 # If the snake encloses itself or is in the state of being enclosed
        self.enclosing_percentage = 0.30 # The maximal percentage of the field the snake has to be in for the enclosing reward to take effect
        self.enclosing_topoff = self.field_size**2*0.60 # The score after which no enclosing penalty will be given

        self.updateFieldVariable()

    def reset(self):
        self.apple = random.randint(0, self.field_size**2-1)
        self.snake = [int(self.field_size**2/2)]
        self.prevAction = 2
        self.memory = []
        self.updateFieldVariable()

    def addMemory(self):

        if len(self.memory) == 6:
            self.memory.pop(0)
        self.memory.append(self.field)

    def step(self, action):
        # Evaluate action, detect if it hits the wall or itself
        index = self.getIndexOfAction(action)
        reward = 0

        # If snake hits wall or itself give negative reward
        if index == -1:
            # Check if snake wants to go in opposite direction it's heading
            if (self.prevAction == 0 and action == 2) or (self.prevAction == 2 and action == 0) or (self.prevAction == 1 and action == 3) or (self.prevAction == 3 and action == 1):
                if self.reward_opposite_dir_topoff < len(self.snake):
                    reward = self.reward_opposite_dir
                index = self.getIndexOfAction(self.prevAction)
                if index == -1:
                    return True, self.reward_death, self.getState(action)
            else:
                return True, self.reward_death, self.getState(action)


        reward += self.getEnclosedPenalty(action)

        # Snake has eaten an apple
        if index == self.apple:
            self.snake.append(index)
            while self.apple in self.snake:
                self.apple = random.randint(0, self.field_size**2-1)
            reward += self.reward_apple
        else:
            if self.gotCloserCheck(action):
                reward += self.reward_closer
            else:
                reward += self.reward_further
            self.snake.append(index)
            self.snake.pop(0)
        
        self.updateFieldVariable()
        
        if self.field in self.memory:
            reward += self.reward_repetitive

        self.addMemory()

        
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

    # def isEnclosedCheck(self):
    #     '''isEnclosedCheck function checks if the snake is currently enclosed and returns it as a boolean.'''
    #     # To save resources only check wether the snake is enclosed if it is close to a object
    #     if self.snakeIsSurrounded:
    #         listOf
    #         for i in self.snake:

    #     else:
    #         return False

    def snakeIsSurrounded(self):
        '''snakeIsSurrounded checks if the snake is surrounded by a block or not'''
        isSurrounded = False
        for i in range(0,4):
            if self.getIndexOfAction(i) == -1:
                isSurrounded = True
        return isSurrounded

    def getEnclosedPenalty(self, action):
        '''getEnclosedPenalty returns reward_enclosing if the snake is about to enclose itself'''
        if len(self.snake) < self.enclosing_topoff:
            index = self.getIndexOfAction(action)
            A = self.getSurface(index)
            Atot = self.field_size**2
            if A <= Atot*self.enclosing_percentage:
                return self.reward_enclosing
            else:
                return 0
        else:
            return 0

    def getSurface(self, index):
        '''getSurface returns empty space of index'''
        if self.field[index] == 0:
            field = np.array(self.field).reshape(self.field_size,self.field_size)
            indexConverted = (index//self.field_size, index%self.field_size)
            field[self.apple//self.field_size][self.apple%self.field_size] = 0
            flood.fill(field, indexConverted, -1)
            return np.count_nonzero(field == -1)
        else:
            return self.field_size**2

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
            fieldSnake = [0]*(self.field_size**2)
            fieldApple = [0]*(self.field_size**2)
            for i in self.snake:
                fieldSnake[i] = 1
            fieldApple[self.apple] = 1
            return fieldSnake+fieldApple
        elif self.mode == 2:
            fieldSnake = [0]*(self.field_size**2)
            fieldApple = [0]*(self.field_size**2)
            for i in self.snake:
                fieldSnake[i] = 1
            fieldApple[self.apple] = 1
            return self.centerSnakeHead(fieldSnake, 1)+self.centerSnakeHead(fieldApple, 0)

    def centerSnakeHead(self, field, fillingVar):
        snakeHead = self.snake[-1]
        headInRow = snakeHead//self.field_size
        headInColumn = snakeHead-(headInRow*self.field_size)

        newField = field

        middle = self.field_size//2
        paddingLeft = -(headInRow-middle)
        paddingTop = -(headInColumn-middle)

        newField = np.array(newField).reshape(self.field_size,self.field_size)
        newField = np.roll(newField,paddingTop,axis=1)
        newField = np.roll(newField,paddingLeft,axis=0)

        stepLeft = 1
        beginLeft = 0
        stepTop = 1
        beginTop = 0
        if paddingLeft < 0:
            stepLeft = -1
            beginLeft = -1
        if paddingTop < 0:
            stepTop = -1
            beginTop = -1

        for i in range(beginLeft,paddingLeft,stepLeft):
            newField[i,:] = fillingVar
        
        for i in range(beginTop,paddingTop,stepTop):
            newField[:,i] = fillingVar
        
        newField = np.delete(newField,0,0)
        newField = np.delete(newField,0,1)
        
        newField = np.array(newField).reshape((middle*2-1)**2)
        return newField.tolist()

if __name__ == '__main__':
    # This code block will only run if you directly run games.py
    
    # If you want to run train_dqn.py when running this file (so switching is not required) set this to true
    trainWhenRun = True

    if trainWhenRun:
        module = __import__('train_dqn')
        train_dqn = getattr(module, 'train_dqn')()
        train_dqn.main(False)
    else:
        g = space_invader()
        g.figures_set([10,10],1)
        g.print()
