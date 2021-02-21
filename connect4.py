# Thanks to floriangardin/connect4-mcts for the main structure of the MCTS (Monte-Carlo tree search) 
# This is part of the mcts file and provided by the floriangardin/connect4-mcts GitHub repo.
import numpy as np

def create_grid(sizeX=6, sizeY=7):
    return np.zeros((sizeX, sizeY), dtype=int)

def reset(grid):
    return np.zeros(grid.shape, dtype=int)

def play(grid_, column, player=None):
    """
    Play at given column, if no player provided, calculate which player must play, otherwise force player to play
    Return new grid and winner
    """
    grid = grid_.copy()
    if player is None:
        player = get_player_to_play(grid)

    if can_play(grid, column):
        row = grid.shape[0] - 1 - np.sum(np.abs(grid[:, column]), dtype=int)
        grid[row, column] = player
    else:
        raise Exception('Error : Column {} is full'.format(column))
    return grid, player if has_won(grid, player, row, column) else 0

def can_play(grid, column):
    """
    Check if the given column is free
    """
    return np.sum(np.abs(grid[:, column])) < len(grid[:, column])

def valid_move(grid):
    return [i for i in range(grid.shape[1]) if can_play(grid, i)]

def has_won(grid, player, row, column):
    """
    Check if player has won with is new piece
    """
    player += 1
    grid += 1
    row_str = ''.join(grid[row, :].astype(str).tolist())
    col_str = ''.join(grid[:, column].astype(str).tolist())
    up_diag_str = ''.join(np.diagonal(grid, offset=(column - row)).astype(str).tolist())
    down_diag_str = ''.join(np.diagonal(np.rot90(grid), offset=-grid.shape[1] + (column + row) + 1).astype(str).tolist())

    grid -= 1
    victory_pattern = str(player)*4
    if victory_pattern in row_str:
        return True
    if victory_pattern in col_str:
        return True
    if victory_pattern in up_diag_str:
        return True
    if victory_pattern in down_diag_str:
        return True

    return False

def get_player_to_play(grid):
    """
    Get player to play given a grid
    """
    player_1 = 0.5 * np.abs(np.sum(grid-1))
    player_2 = 0.5 * np.sum(grid + 1)

    if player_1 > player_2:
        return 1
    else:
        return -1


def to_state(grid):
    grid += 1
    res = ''.join(grid.astype(str).flatten().tolist())
    grid -=1
    return res