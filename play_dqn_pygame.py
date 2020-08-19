import pygame 
from pygame import gfxdraw
import numpy as np
import tensorflow as tf
import os
import datetime
from statistics import mean
import random
import log
import glob
import os
import games as g
import dqn as dqn

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
                print(path)
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


    # SCREEN FUNCTIONS: Functions which display certain scenes

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
            self.first = False
            # (Re)set Variables for TTT
            self.state = [0,0,0,0,0,0,0,0,0]
            self.tictactoe = g.tictactoe()
            self.activePlayer = 0
            self.illegalmove = False
            self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie

            # Initialize DQN
            state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = self.tictactoe.variables

            self.tictactoeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
            self.first = True
            model_name = self.ModelMenu()
            directory = "tictactoe/models/"+model_name+"/TrainNet/"
            tf.saved_model.load(directory)

            self.first = False

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
            if self.illegalmove:
                action = random.randint(0,8)
            else:
                action = self.tictactoeDQN.get_action(self.state, 0)
        
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

    def ticTacToeAvA(self):
        if self.first:
            self.first = False
            # (Re)set Variables for TTT
            self.state = [0,0,0,0,0,0,0,0,0]
            self.tictactoe = g.tictactoe()
            self.activePlayer = 0
            self.illegalmove = False
            self.won = -1 # -1: Game in progress, 0: Cross won, 1:  Circle won, 3: Tie

            # Initialize DQN
            print("Initialized DQN!")
            state, gamma, copy_step, num_states, num_actions, hidden_units, max_experiences, min_experiences, batch_size, alpha, epsilon, min_epsilon, decay = self.tictactoe.variables

            self.tictactoeDQN = dqn.DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, alpha)
            self.first = True
            model_name = self.ModelMenu()
            print(model_name)
            directory = "tictactoe/models/"+model_name+"/TrainNet/"
            tf.saved_model.load(directory)
            
            self.counter = 0

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
            if self.illegalmove:
                action = random.randint(0,8)
            else:
                action = self.tictactoeDQN.get_action(self.state, 0)
        
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
    
                self.addButton(str(item[(3*(self.page+1))-3][0]), 400, 200, 550, 40, item[(3*(self.page+1))-3][1])
                try:
                    self.addButton(str(item[(3*(self.page+1))-2][0]), 400, 300, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    print('EOF')
                try:
                    self.addButton(str(item[(3*(self.page+1))-1][0]), 400, 400, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    print('EOF')
                checkpage = page
            self.addText("Page "+str(self.page+1)+' of '+str(amount_pages), self.ailerons, 15, self.White, 400, 500)
            self.addButton("Back", 70 ,565, 100, 30, self.back)
        else:
            if self.checkpage != self.page:
    
                self.addButtonCallBack(str(item[(3*(self.page+1))-3][0]), 400, 200, 550, 40, item[(3*(self.page+1))-3][1])
                try:
                    self.addButtonCallBack(str(item[(3*(self.page+1))-2][0]), 400, 300, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    print('EOF')
                try:
                    self.addButtonCallBack(str(item[(3*(self.page+1))-1][0]), 400, 400, 550, 40, item[(3*(self.page+1))-3][1])
                except IndexError:
                    print('EOF')
                checkpage = self.page
            self.addText("Page "+str(self.page+1)+' of '+str(amount_pages), self.ailerons, 15, self.White, 400, 500)
            self.addButton("Back", 70 ,565, 100, 30, self.back)
    def ticTacToeMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        self.addText("Choose a mode.", self.blanka, 60, self.White, 400, 100)
        self.addButton("Player vs Player", 400, 200, 400, 40, self.ticTacToePvP)
        self.addButton("Player vs AI", 400, 300, 400, 40, self.ticTacToePvA)
        self.addButton("AI vs AI", 400, 400, 400, 40, self.ticTacToeAvA)
        self.addButton("Back", 70 ,565, 100, 30, self.back)
    
    def ModelMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        models = os.listdir(r'tictactoe/models')
        #models = ['1','2','3','4','5']
        MatModel = []
        for i in range(len(models)):
            MatModel.append([models[i],self.back])
        return(self.scrollBar(0,MatModel, 'x'))

    def scrollbarMenu(self):
        self.scrollBar(0,[['1',self.back],['2',self.back],['3',self.back],['4',self.back],['5',self.back]],'x')

    def startMenu(self):
        # Clear screen and set background color
        self.screen.fill(self.Black)
        
        # Add necessary buttons
        self.addText("G A M E   T A R S", self.anurati, 80, self.White, 400, 100)
        #self.addText("The AI that can play games.", self.ailerons, 30, self.White, 400, 180)
        self.addButton("Tic Tac Toe", 400, 300, 400, 40, self.ticTacToeMenu)
        self.addButton("TEST", 400, 350, 400, 40, self.ModelMenu)
        #self.addButton('Tetris (Work in Progress)', 400, 350, 400, 40, self.ticTacToeMenu)
        self.addButton('Snake (Work in Progress)', 400, 400, 400, 40, self.ticTacToeMenu)

    def back(self):
        self.first = True
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
                s = pygame.Surface((800,600), pygame.SRCALPHA)   # per-pixel alpha
                s.fill((0,0,0,fadeInCounter))                         # notice the alpha value in the color
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