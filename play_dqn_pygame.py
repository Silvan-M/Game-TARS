import pygame 
from pygame import gfxdraw
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

class play_dqn_pygame:
    def __init__(self):
        # Colors
        self.Black = (0, 0, 0)
        self.Grey = (10, 10, 10)
        self.White = (255, 255, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.lightRed = (255, 51, 51)
        self.Blue = (0, 0, 255)
        self.Teal = (74, 255, 231)
        self.TealFaded = (16, 57, 51)

        # Fonts
        self.ailerons = "resources/Ailerons-Regular.otf"
        self.blanka = "resources/Azonix.otf"
        self.anurati = "resources/Anurati-Regular.otf"
        
        # Initialize the current "screen function", this is a function which gets called every frame, depending on the situation it will be a different function, this variable defines this function
        self.currentScreenFunction = None
        self.mouseDidPress = False # This variable will prevent clicking a button twice

        # Variable for checking if first time of displaying a scene
        self.first = True
        # This is a nasty workaround, please don't tell anyone
        self.reallyFirst = True

    # DISPLAY FUNCTIONS: Frequently used structures will be defined here as a function

    def addButton(self, message,x_center,y_center,w,h,action=None):
        mouse = pygame.mouse.get_pos()
        x = x_center - 0.5*w
        y = y_center - 0.5*h

        # Detect mouse hover
        if y < mouse[1] < y+h and x < mouse[0] < x+w:
            # Detect mouse press
            if self.mouseDidPress and action != None:
                if message == "Back" or message == "Menu":
                    # If backbutton, display click effect lightRed
                    pygame.draw.rect(self.screen, self.lightRed, [x, y, w, h])
                elif message == 'Previous': 
                    self.page -=1
                elif  message == 'Next':
                    self.page +=1
                else:
                    # display click effect teal
                    pygame.draw.rect(self.screen, self.Teal, [x, y, w, h])
                if message != 'Previous' and message != 'Next':
                    self.currentScreenFunction = action
                self.mouseDidPress = False
            else:
                # Hover effect
                if message == "Back" or message == "Menu" or message == "Previous":
                    # If backbutton, display hoverd effect lightRed
                    pygame.draw.rect(self.screen, self.lightRed, [x, y, w, h], 2)
                else:
                    pygame.draw.rect(self.screen, self.White, [x, y, w, h], 2)
        else:
            pygame.draw.rect(self.screen, self.Teal, [x, y, w, h], 2)
        
        font = pygame.font.Font("resources/Ailerons-Regular.otf", 20)
        text = font.render(message, True, self.White)
        rect = text.get_rect()
        rect.center = (x+w/2, y+h/2)
        self.screen.blit(text,rect)

    def addButtonCallBack(self, message,x_center,y_center,w,h,path = None):
        mouse = pygame.mouse.get_pos()
        x = x_center - 0.5*w
        y = y_center - 0.5*h

        # Detect mouse hover
        if y < mouse[1] < y+h and x < mouse[0] < x+w:
            # Detect mouse press
            if self.mouseDidPress and path != None:
                pygame.draw.rect(self.screen, self.lightRed, [x, y, w, h])
                self.mouseDidPress = False
                return(path)
            else:
                # Hover effect
                    pygame.draw.rect(self.screen, self.White, [x, y, w, h], 2)
        else:
            pygame.draw.rect(self.screen, self.Teal, [x, y, w, h], 2)
        
        font = pygame.font.Font("resources/Ailerons-Regular.otf", 20)
        text = font.render(message, True, self.White)
        rect = text.get_rect()
        rect.center = (x+w/2, y+h/2)
        self.screen.blit(text,rect)    

    def addText(self, message, fontstring, size, color, x, y):
        font = pygame.font.Font(fontstring, size)
        text = font.render(message, True, color)
        rect = text.get_rect()
        rect.center = (x, y)
        self.screen.blit(text,rect)



    # Tic Tac Toe specific functions
    def drawCross(self, xy, size, width, color):
        x = xy[0]
        y = xy[1]
        r = size/2
        l1 = ((x-r,y-r),(x+r,y+r))
        l2 = ((x-r,y+r),(x+r,y-r))

        pygame.draw.line(self.screen, color, l1[0], l1[1], width)
        pygame.draw.line(self.screen, color, l2[0], l2[1], width)
    
    def drawBoard(self):
        # Draw board lines
        boardcoords = (((175,250),(625,250)),((175,400),(625,400)),((325,100),(325,550)),((475,100),(475,550)))
        for coord in boardcoords:
            pygame.draw.line(self.screen, self.Teal, coord[0], coord[1], 4)

        # Draw the crosses and circles
        symbolcoords = ((250,175),(400,175),(550,175),
                        (250,325),(400,325),(550,325),
                        (250,475),(400,475),(550,475))
        for nr in range(len(self.state)):
            if self.state[nr] == 1:
                # Draw Cross
                self.drawCross(symbolcoords[nr],100,15,self.White)
            elif self.state[nr] == 2:
                # Draw Circle (antialiased, workaround)
                # First draw a filled white circle, then draw a antialiased circle of same radius
                pygame.gfxdraw.filled_circle(self.screen,symbolcoords[nr][0],symbolcoords[nr][1],55,self.White)
                pygame.gfxdraw.aacircle(self.screen,symbolcoords[nr][0],symbolcoords[nr][1],55,self.White)

                # Secondly draw a filled black (background color) circle, then draw a antialiased circle of same radius
                pygame.gfxdraw.filled_circle(self.screen,symbolcoords[nr][0],symbolcoords[nr][1],45,self.Black)
                pygame.gfxdraw.aacircle(self.screen,symbolcoords[nr][0],symbolcoords[nr][1],45,self.Black)

    def checkForTouch(self):
        mouse = pygame.mouse.get_pos()
        touchcoords = ((175,325),(100,250)),((325,475),(100,250)),((475,625),(100,250)),((175,325),(250,400)),((325,475),(250,400)),((475,625),(250,400)),((175,325),(400,550)),((325,475),(400,550)),((475,625),(400,550)) # Coordinate Buttons: ((leftmost x, rightmost, x),(upmost y, downmost y))
        for i in range(len(touchcoords)):
            coords = touchcoords[i]
            if coords[1][0] < mouse[1] < coords[1][1] and coords[0][0] < mouse[0] < coords[0][1]:
                if self.mouseDidPress:
                    self.mouseDidPress = False
                    return i
                self.highlightSquare(i)
        return -1

    def highlightSquare(self, field):
        mouse = pygame.mouse.get_pos()
        touchcoords = ((175,325),(100,250)),((325,475),(100,250)),((475,625),(100,250)),((175,325),(250,400)),((325,475),(250,400)),((475,625),(250,400)),((175,325),(400,550)),((325,475),(400,550)),((475,625),(400,550)) # Coordinate Buttons: ((leftmost x, rightmost, x),(upmost y, downmost y))
        pygame.draw.rect(self.screen, self.TealFaded, (touchcoords[field][0][0],touchcoords[field][1][0],150,150), 0)

    # Snake specific functions
    def drawSnake(self):
        fs = self.snake.field_size
        pygame.draw.rect(self.screen, self.White, [150, 50, 500, 500], 3)
        self.addText("Score: "+str(self.score), self.ailerons, 25, self.White, 400, 25)

        for i, v in enumerate(self.field):
            if v == 1:
                distanceTopWall = i//self.snake.field_size
                distanceLeftWall = i-distanceTopWall*self.snake.field_size
                pygame.draw.rect(self.screen, self.White, [150+500/fs*distanceLeftWall, 50+500/fs*distanceTopWall, 500/fs, 500/fs])
            elif v == 2:
                distanceTopWall = i//self.snake.field_size
                distanceLeftWall = i-distanceTopWall*self.snake.field_size
                pygame.draw.rect(self.screen, self.Teal, [150+500/fs*distanceLeftWall, 50+500/fs*distanceTopWall, 500/fs, 500/fs])


    # SCREEN FUNCTIONS: Functions which display certain scenes
    
    # Tic Tac Toe Screen Functions
    def endGame(self):
        self.screen.fill(self.Black)
        msg = "It's a tie!"
        if self.won == 0:
            msg = "Cross won!"
        elif self.won == 1:
            msg = "Circle won!"
        self.addText(msg, self.blanka, 100, self.White, 400, 200)
        self.addButton("Play again", 400, 300, 400, 40, self.previousGame)
        self.addButton("Menu", 400, 400, 400, 40, self.back)
        self.first = True
        self.reallyFirst = True


    def ticTacToePvP(self):
        if self.first:
            self.first = False
            # (Re)set Variables for TTT
            self.state = [0,0,0,0,0,0,0,0,0]
            self.tictactoe = g.tictactoe()
            self.activePlayer = 0
            self.illegalmove = False
            self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie

        # Clear screen and set background color
        self.screen.fill(self.Black)
        action = self.checkForTouch()
        msg = "Cross's turn!"
        if self.illegalmove:
            if self.activePlayer == 1:
                msg = "Cross: Illegal move! Try again!"
            else:
                msg = "Circle: Illegal move! Try again!"
        elif self.activePlayer == 1:
            msg = "Circle's turn!"

        self.addButton("Back", 70 ,565, 100, 30, self.back)
        self.addText(msg, self.blanka, 30, self.White, 400, 50)
        self.drawBoard()
        if action != -1:
            output = self.tictactoe.step_once(action,self.activePlayer)
            self.state = output[0]
            self.illegalmove = output[5]
            self.activePlayer = output[6]

            if output[3]:
                # Cross won
                self.won = 0
                self.previousGame = self.ticTacToePvP
                self.currentScreenFunction = self.endGame
            elif output[4]:
                # Circle won
                self.won = 1
                self.previousGame = self.ticTacToePvP
                self.currentScreenFunction = self.endGame
            elif output[2]:
                # Tie
                self.won = 2
                self.previousGame = self.ticTacToePvP
                self.currentScreenFunction = self.endGame


    def ticTacToePvA(self):
        if self.first:
            if self.reallyFirst:
                self.first = False
                # (Re)set Variables for TTT
                self.state = [0,0,0,0,0,0,0,0,0]
                self.tictactoe = g.tictactoe()
                self.activePlayer = 0
                self.illegalmove = False
                self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie

                # Initialize DQN
                state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = self.tictactoe.variables

                if not self.modelLoaded:
                    self.tictactoeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)

                self.subpath = "tictactoe"
            if self.modelLoaded == False:
                self.first = self.reallyFirst
                model_name = self.ModelMenu()
                self.first = True
                if model_name:
                    directory = "tictactoe/models/"+model_name+"/TrainNet/"
                    self.tictactoeDQN.model = tf.saved_model.load(directory)
                    self.modelLoaded = True
                    self.first = False
                self.counter = 0
            else:
                self.first = False
            self.reallyFirst = False

        else:
            # Clear screen and set background color
            self.screen.fill(self.Black)
            action = -1
            if self.activePlayer == 1:
                action = self.checkForTouch()

            msg = "AI's turn!"
            if self.illegalmove:
                if self.activePlayer == 1:
                    msg = "Cross: Illegal move! Try again!"
                else:
                    msg = "Circle: Illegal move! Random action!"
            elif self.activePlayer == 1:
                msg = "Circle's turn!"

            self.addButton("Back", 70 ,565, 100, 30, self.back)
            self.addText(msg, self.blanka, 30, self.White, 400, 50)
            self.drawBoard()

            if self.activePlayer == 0:
                randMove, q = self.tictactoeDQN.get_q(np.array(self.tictactoe.convert0neHot(self.state)), 0) # TrainNet determines favorable action
                action = 0
                
                if not randMove:
                    q_list_prob=[]
                    q_list_min = np.min(q)
                    q_list_max = np.max(q)
                    for qi in q:
                        q_list_prob.append(float((qi-q_list_min)/(q_list_max-q_list_min)))
                    for i, p in enumerate(q_list_prob):
                        if self.tictactoe.isIllegalMove(i):
                            q_list_prob[i] = - 1
                    action = np.argmax(q_list_prob)
                    
                else:
                    action = q
            
            if action != -1:
                output = self.tictactoe.step_once(action,self.activePlayer)
                
                self.state = output[0]
                self.illegalmove = output[5]
                self.activePlayer = output[6]

                if output[3]:
                    # Cross won
                    self.won = 0
                    self.previousGame = self.ticTacToePvA
                    self.currentScreenFunction = self.endGame
                elif output[4]:
                    # Circle won
                    self.won = 1
                    self.previousGame = self.ticTacToePvA
                    self.currentScreenFunction = self.endGame
                elif output[2]:
                    # Tie
                    self.won = 2
                    self.previousGame = self.ticTacToePvA
                    self.currentScreenFunction = self.endGame

    def ticTacToePvM(self):
        if self.first:
            self.first = False
            # (Re)set Variables for TTT
            self.state = [0,0,0,0,0,0,0,0,0]
            self.tictactoe = g.tictactoe()
            self.activePlayer = 0
            self.illegalmove = False
            self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie
        # Clear screen and set background color
        self.screen.fill(self.Black)
        action = -1
        if self.activePlayer == 1:
            action = self.checkForTouch()

        msg = "MinMaxs turn!"
        if self.illegalmove:
            if self.activePlayer == 1:
                msg = "Cross: Illegal move! Try again!"
            else:
                msg = "Circle: Illegal move! Random action!"
        elif self.activePlayer == 1:
            msg = "Circle's turn!"

        self.addButton("Back", 70 ,565, 100, 30, self.back)
        self.addText(msg, self.blanka, 30, self.White, 400, 50)
        self.drawBoard()

        if self.activePlayer == 0:
            action = mma.GetMove(self.state, False) # MinMax determines favorable action
            
        
        if action != -1:
            output = self.tictactoe.step_once(action,self.activePlayer)
            
            self.state = output[0]
            self.illegalmove = output[5]
            self.activePlayer = output[6]

            if output[3]:
                # Cross won
                self.won = 0
                self.previousGame = self.ticTacToePvM
                self.currentScreenFunction = self.endGame
            elif output[4]:
                # Circle won
                self.won = 1
                self.previousGame = self.ticTacToePvM
                self.currentScreenFunction = self.endGame
            elif output[2]:
                # Tie
                self.won = 2
                self.previousGame = self.ticTacToePvM
                self.currentScreenFunction = self.endGame

    def ticTacToeAvA(self):
        if self.first:
            if self.reallyFirst:
                # (Re)set Variables for TTT
                self.state = [0,0,0,0,0,0,0,0,0]
                self.tictactoe = g.tictactoe()
                self.activePlayer = 0
                self.illegalmove = False
                self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie

                # Initialize DQN
                state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = self.tictactoe.variables

                if not self.modelLoaded:
                    self.tictactoeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
                
                self.subpath = "tictactoe"
            self.first = self.reallyFirst
            model_name = self.ModelMenu()
            self.first = True
            
            if self.modelLoaded == False:
                if model_name:
                    directory = "tictactoe/models/"+model_name+"/TrainNet/"
                    self.tictactoeDQN.model = tf.saved_model.load(directory)
                    self.first = False
                    self.modelLoaded = True
                self.counter = 0
            else:
                self.first = False
            self.reallyFirst = False
        else:
            self.counter += 1
            # Clear screen and set background color
            self.screen.fill(self.Black)
            action = -1
            msg = "Cross's turn!"
            if self.illegalmove:
                if self.activePlayer == 1:
                    msg = "Cross: Illegal move! Random action!"
                else:
                    msg = "Circle: Illegal move! Random action!"
            elif self.activePlayer == 1:
                msg = "Circle's turn!"

            self.addButton("Back", 70 ,565, 100, 30, self.back)
            self.addText(msg, self.blanka, 30, self.White, 400, 50)
            self.drawBoard()
            
            # Slow down AI to only take action every 30 frames (1 sec), otherwise you can't observe the game
            if self.counter % 30 == 0:
                randMove, q = self.tictactoeDQN.get_q(np.array(self.tictactoe.convert0neHot(self.state)), 0) # TrainNet determines favorable action
                action = 0
                
                if not randMove:
                    q_list_prob=[]
                    q_list_min = np.min(q)
                    q_list_max = np.max(q)
                    for qi in q:
                        q_list_prob.append(float((qi-q_list_min)/(q_list_max-q_list_min)))
                    for i, p in enumerate(q_list_prob):
                        if self.tictactoe.isIllegalMove(i):
                            q_list_prob[i] = - 1
                    action = np.argmax(q_list_prob)
                    
                else:
                    action = q
            
            if action != -1:
                output = self.tictactoe.step_once(action,self.activePlayer)
                
                self.state = output[0]
                self.illegalmove = output[5]
                self.activePlayer = output[6]

                if output[3]:
                    # Cross won
                    self.won = 0
                    self.previousGame = self.ticTacToeAvA
                    self.currentScreenFunction = self.endGame
                elif output[4]:
                    # Circle won
                    self.won = 1
                    self.previousGame = self.ticTacToeAvA
                    self.currentScreenFunction = self.endGame
                elif output[2]:
                    # Tie
                    self.won = 2
                    self.previousGame = self.ticTacToeAvA
                    self.currentScreenFunction = self.endGame
    
    def ticTacToeMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        self.modelLoaded = False
        self.addText("Choose a mode.", self.blanka, 60, self.White, 400, 100)
        self.addButton("Player vs Player", 400, 200, 400, 40, self.ticTacToePvP)
        self.addButton("Player vs AI", 400, 300, 400, 40, self.ticTacToePvA)
        self.addButton("Player vs MinMax", 400, 400, 400, 40, self.ticTacToePvM)
        self.addButton("AI vs AI", 400, 500, 400, 40, self.ticTacToeAvA)
        self.addButton("Back", 70 ,565, 100, 30, self.back)
    
    # Snake Screen Functions
    def endSnake(self):
        self.screen.fill(self.Black)
        msg = "Score: "+str(self.score)
        self.addText(msg, self.blanka, 100, self.White, 400, 200)
        self.addButton("Play again", 400, 300, 400, 40, self.previousGame)
        self.addButton("Menu", 400, 400, 400, 40, self.back)
        self.first = True
        self.reallyFirst = True
    
    def snakeP(self):
        if self.first:
            self.first = False
            self.snake = g.snake()
            self.field = self.snake.field
            self.counter = 0
            self.action = 0
            self.score = 0

        self.counter += 1
        # Clear screen and set background color
        self.screen.fill(self.Black)

        self.addButton("Back", 70 ,565, 100, 30, self.back)
        self.drawSnake()


        move_ticker = 0
        keys=pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if move_ticker == 0:
                move_ticker = 10
                self.action = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if move_ticker == 0:   
                move_ticker = 10     
                self.action = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            if move_ticker == 0:   
                move_ticker = 10     
                self.action = 0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if move_ticker == 0:   
                move_ticker = 10     
                self.action = 2
        

        if self.action != -1 and (self.counter % 5 == 0):
            done, reward, observations =  self.snake.step(self.action)
            self.field = self.snake.field

            if reward == self.snake.reward_apple:
                self.score += 1

            if done:
                self.won = 0
                self.previousGame = self.snakeP
                self.currentScreenFunction = self.endSnake

    def abortAndRetry(self):
        self.currentScreenFunction = self.previousGame
        self.first = True
        self.reallyFirst = True

    def snakeAI(self):
        if self.first:
            if self.reallyFirst:
                self.first = False
                self.snake = g.snake()
                self.field = self.snake.field
                self.counter = 0
                self.action = 0
                self.score = 0

                # Initialize DQN
                state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = self.snake.variables
                self.state = state

                if not self.modelLoaded:
                    self.snakeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
                
                self.subpath = "snake"
            self.first = self.reallyFirst
            model_name = self.ModelMenu()
            self.first = True

            # Is set for the abort and retry function
            self.previousGame = self.snakeAI
            
            if self.modelLoaded == False:
                if model_name:
                    directory = "snake/models/"+model_name+"/TrainNet/"
                    self.snakeDQN.model = tf.saved_model.load(directory)
                    self.first = False
                    self.modelLoaded = True
                self.counter = 0
            else:
                self.first = False
            self.reallyFirst = False
        else:
            self.counter += 1
            # Clear screen and set background color
            self.screen.fill(self.Black)

            self.addButton("Back", 120 ,570, 200, 30, self.back)
            self.addButton("Abort and Retry", 680 ,570, 200, 30, self.abortAndRetry)
            self.drawSnake()

            if self.action != -1 and (self.counter % 5 == 0):
                self.action = self.snakeDQN.get_action(np.array(self.state),0)
                done, reward, state =  self.snake.step(self.action)
                self.field = self.snake.field
                self.state = state

                if reward == self.snake.reward_apple:
                    self.score += 1

                if done:
                    self.won = 0
                    self.previousGame = self.snakeAI
                    self.currentScreenFunction = self.endSnake

    def snakeMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        self.modelLoaded = False
        self.addText("Who should play?", self.blanka, 60, self.White, 400, 100)
        self.addButton("Player", 400, 250, 400, 40, self.snakeP)
        self.addButton("AI", 400, 350, 400, 40, self.snakeAI)
        self.addButton("Back", 70 ,565, 100, 30, self.back)
    # SpaceInvader screen function

    def spaceInvaderMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        self.modelLoaded = False
        self.addText("Who should play?", self.blanka, 60, self.White, 400, 100)
        self.addButton("Player", 400, 250, 400, 40, self.spaceInvaderP)
        self.addButton("AI", 400, 350, 400, 40, self.back)
        self.addButton("Back", 70 ,565, 100, 30, self.back)

    def spaceInvaderP(self):
        # when changing dimensions choose values so that all corresponding calculations have integer solutions
        # if first draw, set variables
        if self.first:
            self.first = False
            # load data from the game
            self.spaceInvader = g.space_invader()
            # calculate the field-size
            self.field = [ self.spaceInvader.lenState,self.spaceInvader.height]
            # calculate ratio
            self.ratio = self.spaceInvader.lenState/self.spaceInvader.height
            self.counter = 0
            self.action = ['N', False]
            # buffer makes game slower
            self.buffer = 0
            self.shoot_buffer = 0
            # internal score
            self.score = self.spaceInvader.score
            self.health = self.spaceInvader.health
            # set width of visualized field
            self.width = 300
            # calculate the value of the visualized field
            self.dimensions = [int(self.ratio * self.width) , int(self.width) ]
            print(self.dimensions)
            print(self.field)
        self.drawSpaceInvader()

    def endSpaceInvader(self):
        self.first = True
        self.screen.fill(self.Black)
        msg = "Score: "+str(self.score[3])
        self.addText(msg, self.blanka, 100, self.White, 400, 200)
        self.addButton("Play again", 400, 300, 400, 40, self.previousGame)
        self.addButton("Menu", 400, 400, 400, 40, self.back)

    def drawSpaceInvader(self):
        self.buffer +=1
        # empty screen
        # calculate x and y positioning coordinates and round to integers
        x_len = self.dimensions[0]/self.field[0] 
        #x_len = int(x_len) 
        y_len = self.dimensions[1]/self.field[1] 
        #y_len = int(y_len)
        self.screen.fill(self.Black)
        # make outer rectangle (white)
        pygame.draw.rect(self.screen, self.White, [400 - self.dimensions[0]/2 - x_len/2, 250 - self.dimensions[1]/2 -y_len/2, self.dimensions[0] + x_len, self.dimensions[1]+ y_len] , 4)
        # add scoreboard
        self.health = self.spaceInvader.health
        # check if health is zero and if so go to next screen
        if self.health <= 0:
            self.previousGame = self.spaceInvaderP
            self.currentScreenFunction = self.endSpaceInvader
            self.endSpaceInvader()
        # iterate over all elements of field
        for x in range (len(self.spaceInvader.state)):
            for y in range(len(self.spaceInvader.state[x])):
                if self.spaceInvader.state[x][y] != 0  :
                    x_coord = 400 - self.dimensions[0]/2 + x*x_len 
                    y_coord = 250 - self.dimensions[1]/2 + y*y_len 
                    pygame.draw.rect(self.screen, self.Teal,[x_coord , y_coord , x_len , y_len])
        # heart symbol
        for i in range(self.health):
            pygame.draw.rect(self.screen, self.Teal,[25 + x_len + (i*30) , 430 - y_len, x_len  , y_len])
            pygame.draw.rect(self.screen, self.Teal,[25 + (x_len *3) + (i*30), 430 - y_len, x_len  , y_len])
            pygame.draw.rect(self.screen, self.Teal,[25 + (i*30), 430, x_len * 5 , y_len])
            pygame.draw.rect(self.screen, self.Teal,[25 + x_len+ (i*30) , 430+y_len, x_len * 3 , y_len])  
            pygame.draw.rect(self.screen, self.Teal,[25  + (x_len*2)+ (i*30) , 430+ (y_len*2), x_len  , y_len])
        
        # load images
        lvl1 = pygame.image.load('resources\lvl1.png')
        lvl2 = pygame.image.load('resources\lvl2.png')
        lvl3 = pygame.image.load('resources\lvl3.png')
        wave = pygame.image.load('resources\wave.png')
        score = pygame.image.load('resources\score.png')
        # transform images
        lvl1 = pygame.transform.scale(lvl1, (int(x_len*15),int( y_len*14)))
        lvl2 = pygame.transform.scale(lvl2, (int(x_len*15),int( y_len*13)))
        lvl3 = pygame.transform.scale(lvl3, (int(x_len*15),int( y_len*14)))
        wave = pygame.transform.scale(wave, (int(x_len*8),int( y_len*6))) 
        score = pygame.transform.scale(score, (int(x_len*15),int( y_len*14))) 
        self.screen.blit(lvl1, (150, 400))
        self.screen.blit(lvl2, (250, 400))
        self.screen.blit(lvl3, (350, 400))
        self.screen.blit(wave, (450, 418))
        self.screen.blit(score, (550, 410))
        self.addText(str(self.score[0]), self.ailerons, 25, self.White, 220, 435)
        self.addText(str(self.score[1]), self.ailerons, 25, self.White, 320, 435)
        self.addText(str(self.score[2]), self.ailerons, 25, self.White, 420, 435)
        self.addText(str(self.score[4]), self.ailerons, 25, self.White, 520, 435)
        self.addText(str(self.score[3]), self.ailerons, 25, self.White, 670, 435)
        keys=pygame.key.get_pressed()
        self.shoot_buffer += 1
        self.action = ['N', False]
        move_ticker = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if move_ticker == 0:
                move_ticker = 10
                self.action = ['L', False]
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if move_ticker == 0:   
                move_ticker = 10     
                self.action = ['R', False]
        if keys[pygame.K_UP] or keys[pygame.K_w]: 
            if self.shoot_buffer >= 5:
                if move_ticker == 0:   
                    move_ticker = 10   
                    self.shoot_buffer = 0  
                    self.action = ['N', True]
        if self.buffer % 1 == 0: 
            #print(self.spaceInvader.print())
            # make next step
            self.spaceInvader.step(self.action)
            #print('step called')
            #for i in range(len(self.spaceInvader.state)):
             #   print(self.spaceInvader.state[i])

    def scrollBar(self, page, item, mode):
        if self.first:
            self.first = False
            self.checkpage = -1
            self.item = item
            self.page = page
        amount_pages = len(item)//3
        if len(item)%3 != 0:
            amount_pages += 1
        self.screen.fill(self.Black)
        self.addButton("Back", 70 , 565, 100, 30, self.back)
        if self.page+1 <= amount_pages-1:
            self.addButton('Next', 600, 500, 70, 30, self.page)
        if self.page != 0:
            self.addButton('Previous', 200, 500, 110, 30, self.page)
        if mode != 'model':
            if self.checkpage != self.page:
                callback1 = self.addButtonCallBack(str(item[(3*(self.page+1))-3][0]), 400, 200, 700, 40, str(item[(3*(self.page+1))-3][0]))
                try:
                    callback2 = self.addButtonCallBack(str(item[(3*(self.page+1))-2][0]), 400, 300, 700, 40, str(item[(3*(self.page+1))-2][0]))
                    if callback2:
                        return callback2
                except IndexError:
                    # EOF, do nothing (pass)
                    pass
                try:
                    callback3 = self.addButtonCallBack(str(item[(3*(self.page+1))-1][0]), 400, 400, 700, 40, str(item[(3*(self.page+1))-1][0]))
                    if callback3:
                        return callback3
                except IndexError:
                    # EOF, do nothing (pass)
                    pass
                checkpage = page

                if callback1:
                    return callback1

            self.addText("Page "+str(self.page+1)+' of '+str(amount_pages), self.ailerons, 15, self.White, 400, 500)
            self.addButton("Back", 70 ,565, 100, 30, self.back)
        else:
            if self.checkpage != self.page:
                
                self.addButton(str(item[(3*(self.page+1))-3][0]), 400, 200, 550, 40, item[(3*(self.page+1))-3][1])
                try:
                    self.addButton(str(item[(3*(self.page+1))-2][0]), 400, 300, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    # EOF, do nothing (pass)
                    pass
                try:
                    self.addButton(str(item[(3*(self.page+1))-1][0]), 400, 400, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    # EOF, do nothing (pass)
                    pass
                checkpage = self.page
            self.addText("Page "+str(self.page+1)+' of '+str(amount_pages), self.ailerons, 15, self.White, 400, 500)
            self.addButton("Back", 70 ,565, 100, 30, self.back)
    
    def ModelMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        models = os.listdir(r'{}/models'.format(self.subpath))
        #models = ['1','2','3','4','5']
        MatModel = []
        for i, v in enumerate(models):
            if v != ".DS_Store":
                MatModel.append([v,None])
        MatModel = sorted(MatModel, reverse=True)
        return(self.scrollBar(0,MatModel, 'x'))

    def scrollbarMenu(self):
        self.scrollBar(0,[['1',self.back],['2',self.back],['3',self.back],['4',self.back],['5',self.back]],'x')

    def startMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        
        # Add necessary buttons
        self.addText("G A M E   T A R S", self.anurati, 80, self.White, 400, 100)
        self.addText("The AI that can play games.", self.ailerons, 30, self.White, 400, 180)
        self.addButton("Tic Tac Toe", 400, 300, 400, 40, self.ticTacToeMenu)
        self.addButton('Snake', 400, 350, 400, 40, self.snakeMenu)
        self.addButton("Space Invaders", 400, 400, 400, 40, self.spaceInvaderMenu)

    def back(self):
        self.first = True
        self.reallyFirst = True
        self.currentScreenFunction = self.startMenu

    def run(self):
        pygame.init()

        # Set the height and width of the screen
        self.size = (800, 600)
        self.screen = pygame.display.set_mode(self.size)

        # Set display name
        pygame.display.set_caption("Game-TARS")

        # Setting screen to be the start menu
        self.currentScreenFunction = self.startMenu

        done = False
        self.clock = pygame.time.Clock()

        # WHILE LOOP: Keeps the game running at 30 fps

        fadeInCounter = 255

        while not done:
            for event in pygame.event.get():  # Get user event
                # Stop the game if user exited
                if event.type == pygame.QUIT:  
                    done = True 
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.mouseDidPress:
                        self.mouseDidPress = True
                    else: 
                        self.mouseDidPress = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouseDidPress = False
            
            # Execute the current screen function
            self.currentScreenFunction()
            
            # Make black fade in at the beginning
            if fadeInCounter >= 5:
                fadeInCounter -= 5
                s = pygame.Surface((800,600), pygame.SRCALPHA) # per-pixel alpha
                s.fill((0,0,0,fadeInCounter)) # notice the alpha value in the color
                self.screen.blit(s, (0,0))

            # Update screen
            pygame.display.flip()
            
            # Limit the refresh rate to 30 fps
            self.clock.tick(30)
        
        pygame.quit()

# Initialize a session
session = play_dqn_pygame()
# Run the session and open pygame
session.run()