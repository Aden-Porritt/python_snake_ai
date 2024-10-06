import numpy as np
from numba import njit
from numba.pycc import CC
import numba

cc = CC('snake_eng')

@njit()
@cc.export('make_board', 'Tuple((i8[:, :, :], i8[:, :], i8[:], b1[:, :], b1[:]))(i8[:], i8)')
def make_board(size, number_of_players):
    snakes_lenght = np.array([4 for _ in range(number_of_players)], 'i8')
    board = np.zeros((1 + number_of_players, size[0], size[1]), 'i8')
    for player in range(number_of_players):
        if player % 2 == 0:
            for i in range(4):
                board[1 + player][((player + 1) * size[0] // (number_of_players + 1))][i] = i + 1
        if player % 2 == 1:
            for i in range(4):
                board[1 + player][((player + 1) * size[0] // (number_of_players + 1))][-i - 1] = i + 1

    snakes_pos = np.zeros((number_of_players, 2), 'i8')
    for player in range(number_of_players):
        if player % 2 == 0:
            snakes_pos[player] = np.array([(player + 1) * size[0] // (number_of_players + 1), 3])
        if player % 2 == 1:
            snakes_pos[player] = np.array([(player + 1) * size[0] // (number_of_players + 1), size[1] - 4])

    for _ in range(5):
        board = spawn_apple(board, size)

    wall_spawn = np.ones((size[0], size[1]), '?')
    wall_spawn[1][0] = False
    wall_spawn[0][1] = False
    wall_spawn[-1][1] = False
    wall_spawn[-2][0] = False
    wall_spawn[1][-1] = False
    wall_spawn[0][-2] = False
    wall_spawn[-2][-1] = False
    wall_spawn[-1][-2] = False

    snakes_alive = np.array([True for _ in range(number_of_players)], '?')

    return board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive

@njit()
@cc.export('spawn_apple', 'i8[:, :, :](i8[:, :, :], i8[:])')
def spawn_apple(board, size):
    start_row = np.random.randint(0, size[0] - 1)
    start_col = np.random.randint(0, size[1] - 1)
    for row in range(start_row, size[0] + start_row):
        for col in range(start_col, size[0] + start_col):
            row_ = row % size[0]
            col_ = col % size[1]
            if np.sum(board[:, row_, col_]) == 0:
                board[0][row_][col_] = 1
                return board
    return board

@njit()
@cc.export('spawn_wall', 'Tuple((i8[:, :, :], b1[:, :]))(i8[:, :, :], b1[:, :], i8[:, :], i8[:])')
def spawn_wall(board, wall_spawn_check, snakes_pos, size):
    start_row = np.random.randint(0, size[0] - 1)
    start_col = np.random.randint(0, size[1] - 1)
    number_of_players = len(board) - 1
    for row in range(start_row, size[0] + start_row):
        for col in range(start_col, size[0] + start_col):
            row_ = row % size[0]
            col_ = col % size[1]
            if np.sum(board[:, row_, col_]) == 0:
                if wall_spawn_check[row_][col_]:
                    for player in range(number_of_players):
                        if (snakes_pos[player][0] + 2 >= row_  and snakes_pos[player][0] - 2 <= row_) and (snakes_pos[player][1] + 2 >= col_  and snakes_pos[player][1] - 2 <= col_):
                            break
                    else:
                        board[0][row_][col_] = 2
                        return change_wall_spawn(board, np.array([row_, col_]), wall_spawn_check, size)
    return board, wall_spawn_check

@njit()
@cc.export('change_wall_spawn', 'Tuple((i8[:, :, :], b1[:, :]))(i8[:, :, :], i8[:], b1[:, :], i8[:])')
def change_wall_spawn(board, wall_spawn, wall_spawn_check, size):
    wall_spawn_check[wall_spawn[0]][wall_spawn[1]] = False
    row = wall_spawn[0]
    col = wall_spawn[1]
    if row == 0:
        if col == 0:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
        elif col == size[1] - 1:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
        elif col == 2:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1] - 2] = False
        elif col == size[1] - 3:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1] + 2] = False
        else:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False

    elif row == size[0] - 1:
        if col == 0:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
        elif col == size[1] - 1:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
        elif col == 2:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1] - 2] = False
        elif col == size[1] - 3:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1] + 2] = False
        else:
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 2] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 2] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False

    elif col == 0:
        if row == 2:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1] + 2] = False
        elif row == size[0] - 3:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1] + 2] = False
        else:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
    
    elif col == size[1] - 1:
        if row == 2:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1] - 2] = False
        elif row == size[0] - 3:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1] - 2] = False
        else:
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] + 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 2][wall_spawn[1]] = False
            wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
            wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False

    else:
        wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] + 1] = False
        wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1]] = False
        wall_spawn_check[wall_spawn[0] + 1][wall_spawn[1] - 1] = False
        wall_spawn_check[wall_spawn[0]][wall_spawn[1] + 1] = False
        wall_spawn_check[wall_spawn[0]][wall_spawn[1] - 1] = False
        wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] + 1] = False
        wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1]] = False
        wall_spawn_check[wall_spawn[0] - 1][wall_spawn[1] - 1] = False

    return board, wall_spawn_check

@njit()
@cc.export('move_snake', 'Tuple((i8[:, :, :], i8[:, :], i8[:], b1[:], b1[:, :]))(i8[:], i8[:, :, :], i8[:, :], i8[:], b1[:, :], b1[:], i8[:])')
def move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size):
    board = np.copy(board)
    snakes_pos = np.copy(snakes_pos)
    wall_spawn = np.copy(wall_spawn)
    snakes_lenght = np.copy(snakes_lenght)
    snakes_alive = np.copy(snakes_alive)
    for index, move in enumerate(moves):
        if snakes_alive[index]:
            if move != 4:
                if move == 0:
                    snakes_pos[index][0] += 1
                elif move == 1:
                    snakes_pos[index][1] += 1
                elif move == 2:
                    snakes_pos[index][0] -= 1
                elif move == 3:
                    snakes_pos[index][1] -= 1
                if snakes_pos[index][0] < 0 or snakes_pos[index][0] > size[0] - 1 or snakes_pos[index][1] < 0 or snakes_pos[index][1] > size[1] - 1:
                    snakes_alive[index] = False
                else:
                    if board[0][snakes_pos[index][0]][snakes_pos[index][1]] == 2:
                        snakes_alive[index] = False
                    for i in range(len(moves)):
                        if board[i + 1][snakes_pos[index][0]][snakes_pos[index][1]] > 1:
                            snakes_alive[index] = False
    for index in range(len(moves)):
        if not snakes_alive[index]:
            board[index + 1] *= 0
        else:
            if board[0][snakes_pos[index][0]][snakes_pos[index][1]] == 1:
                snakes_lenght[index] += 1
                if np.sum(snakes_lenght) % 2 == 1:
                    board, wall_spawn = spawn_wall(board, wall_spawn, snakes_pos, size)
            else:
                board[index + 1] -= 1
            board[index + 1][snakes_pos[index][0]][snakes_pos[index][1]] = snakes_lenght[index]
            board[index + 1] = np.clip(board[index + 1], 0, 1000)
    for index in range(len(moves)):
        if not snakes_alive[index]:
            board[index + 1] *= 0
        for i in range(len(moves)):
            if i == index:
                continue
            else:
                if board[i + 1][snakes_pos[index][0]][snakes_pos[index][1]] == snakes_lenght[i]:
                    snakes_alive[index] = False
                    snakes_alive[i] = False
        else:
            if board[0][snakes_pos[index][0]][snakes_pos[index][1]] == 1:
                spawn_apple(board, size)
                board[0][snakes_pos[index][0]][snakes_pos[index][1]] = 0
    return board, snakes_pos, snakes_lenght, snakes_alive, wall_spawn

@njit()
@cc.export('flood_fill_', 'i8(i8[:, :], i8[:], b1, i8[:])')
def flood_fill_(board, snake_pos, find_tail, size):
    board = np.copy(board)
    pos_array = np.zeros((size[0] * size[1], 2), 'i8')
    pos_array[0] = snake_pos 
    pos_start_pointer = 0
    pos_end_pointer = 1
    while pos_start_pointer != pos_end_pointer:
        start_index = pos_start_pointer
        pos_start_pointer = pos_end_pointer
        for index in range(start_index, pos_end_pointer):
            pos = pos_array[index]
            if pos[0] + 1 != size[0]:
                if board[pos[0] + 1][pos[1]] == 0:
                    new_pos = pos + np.array([1, 0], 'i8')
                    board[new_pos[0]][new_pos[1]] = 2
                    pos_array[pos_end_pointer] = new_pos
                    pos_end_pointer += 1
            if pos[1] + 1 != size[1]:
                if board[pos[0]][pos[1] + 1] == 0:
                    new_pos = pos + np.array([0, 1], 'i8')
                    board[new_pos[0]][new_pos[1]] = 2
                    pos_array[pos_end_pointer] = new_pos
                    pos_end_pointer += 1
            if pos[0] != 0:
                if board[pos[0] - 1][pos[1]] == 0:
                    new_pos = pos + np.array([-1, 0], 'i8')
                    board[new_pos[0]][new_pos[1]] = 2
                    pos_array[pos_end_pointer] = new_pos
                    pos_end_pointer += 1
            if pos[1] != 0:
                if board[pos[0]][pos[1] - 1] == 0:
                    new_pos = pos + np.array([0, -1], 'i8')
                    board[new_pos[0]][new_pos[1]] = 2
                    pos_array[pos_end_pointer] = new_pos
                    pos_end_pointer += 1
    return pos_end_pointer

@njit()
@cc.export('two_point_flood_fill', 'i8(i8[:, :], i8[:], i8[:], i8[:])')
def two_point_flood_fill(board, start_pos1, start_pos2, size):
    board = np.copy(board)
    pos1_array = np.zeros((size[0] * size[1], 2), 'i8')
    pos1_array[0] = start_pos1
    pos1_start_pointer = 0
    pos1_end_pointer = 1

    pos2_array = np.zeros((size[0] * size[1], 2), 'i8')
    pos2_array[0] = start_pos2
    pos2_start_pointer = 0
    pos2_end_pointer = 1

    while pos1_start_pointer != pos1_end_pointer or pos2_start_pointer != pos2_end_pointer:
        player_one_board = np.zeros((size[0], size[1]), 'i8')
        start_index = pos1_start_pointer
        pos1_start_pointer = pos1_end_pointer
        for index in range(start_index, pos1_end_pointer):
            pos1 = pos1_array[index]
            if pos1[0] + 1 != size[0]:
                if board[pos1[0] + 1][pos1[1]] == 0 and player_one_board[pos1[0] + 1][pos1[1]] == 0:
                    new_pos1 = pos1 + np.array([1, 0], 'i8')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[1] + 1 != size[1]:
                if board[pos1[0]][pos1[1] + 1] == 0 and player_one_board[pos1[0]][pos1[1] + 1] == 0:
                    new_pos1 = pos1 + np.array([0, 1], 'i8')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[0] != 0:
                if board[pos1[0] - 1][pos1[1]] == 0 and player_one_board[pos1[0] - 1][pos1[1]] == 0:
                    new_pos1 = pos1 + np.array([-1, 0], 'i8')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[1] != 0:
                if board[pos1[0]][pos1[1] - 1] == 0 and player_one_board[pos1[0]][pos1[1] - 1] == 0:
                    new_pos1 = pos1 + np.array([0, -1], 'i8')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1

        player_two_board = np.zeros((size[0], size[1]), 'i8')
        start_index = pos2_start_pointer
        pos2_start_pointer = pos2_end_pointer
        for index in range(start_index, pos2_end_pointer):
            pos2 = pos2_array[index]
            if pos2[0] + 1 != size[0]:
                if board[pos2[0] + 1][pos2[1]] == 0 and player_two_board[pos2[0] + 1][pos2[1]] == 0:
                    new_pos2 = pos2 + np.array([1, 0], 'i8')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[1] + 1 != size[1]:
                if board[pos2[0]][pos2[1] + 1] == 0 and player_two_board[pos2[0]][pos2[1] + 1] == 0:
                    new_pos2 = pos2 + np.array([0, 1], 'i8')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[0] != 0:
                if board[pos2[0] - 1][pos2[1]] == 0 and player_two_board[pos2[0] - 1][pos2[1]] == 0:
                    new_pos2 = pos2 + np.array([-1, 0], 'i8')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[1] != 0:
                if board[pos2[0]][pos2[1] - 1] == 0 and player_two_board[pos2[0]][pos2[1] - 1] == 0:
                    new_pos2 = pos2 + np.array([0, -1], 'i8')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
        board += player_one_board + player_two_board
    return pos1_end_pointer - pos2_end_pointer

@njit()
@cc.export('two_point_flood_fill_eval', 'f8(i8[:, :, :], i8[:, :], i8[:])')
def two_point_flood_fill_eval(board, snakes_pos, size):
    board = np.copy(board)
    board = np.clip((board[0] - 1) * 4 + board[1] + board[2], 0, 1000)
    count = two_point_flood_fill(board, snakes_pos[0], snakes_pos[1], size)
    return count / 4 + 0.0

if __name__ == "__main__":
    cc.compile()