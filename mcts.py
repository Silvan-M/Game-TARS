# Thanks to floriangardin/connect4-mcts for the main structure of the MCTS (Monte-Carlo tree search)
from connect4 import *
import time
import numpy as np
import random

class Node:

    def __init__(self, state, winning, move, parent):
        self.parent = parent
        self.move = move
        self.win = 0
        self.games = 0
        self.children = None
        self.state = state
        self.winner = winning

    def set_children(self, children):
        self.children = children

    def get_uct(self):
        # Upper Confidence bound for Trees, The "value" of a node
        if self.games == 0:
            return None
        return (self.win/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)


    def select_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None, None

        winners = [child for child in self.children if child.winner]
        if len(winners) > 0:
            return winners[0], winners[0].move

        games = [child.win/child.games if child.games > 0 else 0 for child in self.children]
        best_child = self.children[np.argmax(games)]
        return best_child, best_child.move

    def get_children_moves(self):
        moves = []
        for child in self.children:
            moves.append(child.move)
        return moves

    def get_children_with_move(self, move):
        if self.children is None:
            return None
        for child in self.children:
            if child.move == move:
                return child
        raise Exception('Not existing child')

def random_play(grid):
    """
    Play a random game starting by state and player
    Return winner
    """

    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        selected_move = random.choice(moves)
        player_to_play = get_player_to_play(grid)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play

def random_play_improved(grid):

    def get_winning_moves(grid, moves, player):
        return [move for move in moves if play(grid, move, player=player)[1]]

    # If can win, win
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0
        player_to_play = get_player_to_play(grid)

        winning_moves = get_winning_moves(grid, moves, player_to_play)
        loosing_moves = get_winning_moves(grid, moves, -player_to_play)

        if len(winning_moves) > 0:
            selected_move = winning_moves[0]
        elif len(loosing_moves) == 1:
            selected_move = loosing_moves[0]
        else:
            selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return player_to_play

def train_mcts_during(mcts, training_time, TrainNet, TargetNet, train=True):
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts, TrainNet, TargetNet = train_mcts_once(mcts, TrainNet, TargetNet, train)
        current = int(round(time.time() * 1000))
    return mcts, TrainNet, TargetNet

def train_mcts_once(mcts=None, TrainNet=None, TargetNet=None, train=True):

    if mcts is None:
        mcts = Node(create_grid(), 0, None,  None)

    node = mcts

    # selection
    while node.children is not None:
        # Select highest uct
        ucts = [child.get_uct() for child in node.children]
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # expansion, no expansion if terminal node
    moves = valid_move(node.state)

    # Remove unecessary branches using a Deep-Q-Network
    _, prob = TrainNet.get_prob(np.array(node.state), 1) # TrainNet determines favorable action
    for i in range(len(moves)-4):
        moves.pop(np.argmin(prob))
    


    if len(moves) > 0:

        if node.winner == 0:
            states = [(play(node.state, move), move) for move in moves]
            node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])
            # simulation
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play_improved(node.state)
        else:
            victorious = node.winner

        # backpropagation
        parent = node
        iter = 0
        done = True
        while parent is not None:
            pp = parent.parent
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
                


                # Train DQN if enabled
                reward_win = 500
                if train and (pp is not None):
                    exp = {'s': np.array(pp.state), 'a': parent.move, 'r': reward_win, 's2': np.array(parent.state), 'done': done} # make memory callable as a dictionary
                    TrainNet.add_experience(exp) # memorizes experience, if the max amount is exceeded the oldest element gets deleted
                    loss = TrainNet.train(TargetNet) # returns loss 
                    iter += 1 # increment the counter
                    copy_step = 150
                    if iter % copy_step == 0: #copies the weights of the dqn to the TrainNet if the iter is a multiple of copy_step
                        TargetNet.copy_weights(TrainNet)

            parent = pp
            done = False


    else:
        print('no valid moves, expended all')

    return mcts, TrainNet, TargetNet