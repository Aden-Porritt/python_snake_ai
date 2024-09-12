import numpy as np
import time
from numba import njit

@njit()
def make_board(size, number_of_players):
    snakes_lenght = np.array([4 for _ in range(number_of_players)], 'i4')
    board = np.zeros((1 + number_of_players, size[0], size[1]), 'i4')
    for player in range(number_of_players):
        if player % 2 == 0:
            for i in range(4):
                board[1 + player][((player + 1) * size[0] // (number_of_players + 1))][i] = i + 1
        if player % 2 == 1:
            for i in range(4):
                board[1 + player][((player + 1) * size[0] // (number_of_players + 1))][-i - 1] = i + 1

    snakes_pos = np.zeros((number_of_players, 2), 'i4')
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
def flood_fill(board, snake_pos, find_tail, count, size):
    snake_pos = np.copy(snake_pos)
    if snake_pos[0] + 1 != size[0]:
        if board[snake_pos[0] + 1][snake_pos[1]] == 1:
            find_tail = True
        elif board[snake_pos[0] + 1][snake_pos[1]] == 0:
            new_snake_pos = snake_pos + np.array([1, 0])
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[1] + 1 != size[1]:
        if board[snake_pos[0]][snake_pos[1] + 1] == 1:
            find_tail = True
        elif board[snake_pos[0]][snake_pos[1] + 1] == 0:
            new_snake_pos = snake_pos + np.array([0, 1])
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[0] != 0:
        if board[snake_pos[0] - 1][snake_pos[1]] == 1:
            find_tail = True
        elif board[snake_pos[0] - 1][snake_pos[1]] == 0:
            new_snake_pos = snake_pos + np.array([-1, 0])
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[1] != 0:
        if board[snake_pos[0]][snake_pos[1] - 1] == 1:
            find_tail = True
        elif board[snake_pos[0]][snake_pos[1] - 1] == 0:
            new_snake_pos = snake_pos + np.array([0, -1])
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    return find_tail, count

@njit()
def two_point_flood_fill(board, start_pos1, start_pos2, size):
    board = np.copy(board)
    pos1_array = np.zeros((size[0] * size[1], 2), 'i4')
    pos1_array[0] = start_pos1
    pos1_start_pointer = 0
    pos1_end_pointer = 1

    pos2_array = np.zeros((size[0] * size[1], 2), 'i4')
    pos2_array[0] = start_pos2
    pos2_start_pointer = 0
    pos2_end_pointer = 1

    while pos1_start_pointer != pos1_end_pointer or pos2_start_pointer != pos2_end_pointer:
        player_one_board = np.zeros((size[0], size[1]), 'i4')
        start_index = pos1_start_pointer
        pos1_start_pointer = pos1_end_pointer
        for index in range(start_index, pos1_end_pointer):
            pos1 = pos1_array[index]
            if pos1[0] + 1 != size[0]:
                if board[pos1[0] + 1][pos1[1]] == 0 and player_one_board[pos1[0] + 1][pos1[1]] == 0:
                    new_pos1 = pos1 + np.array([1, 0], 'i4')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[1] + 1 != size[1]:
                if board[pos1[0]][pos1[1] + 1] == 0 and player_one_board[pos1[0]][pos1[1] + 1] == 0:
                    new_pos1 = pos1 + np.array([0, 1], 'i4')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[0] != 0:
                if board[pos1[0] - 1][pos1[1]] == 0 and player_one_board[pos1[0] - 1][pos1[1]] == 0:
                    new_pos1 = pos1 + np.array([-1, 0], 'i4')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1
            if pos1[1] != 0:
                if board[pos1[0]][pos1[1] - 1] == 0 and player_one_board[pos1[0]][pos1[1] - 1] == 0:
                    new_pos1 = pos1 + np.array([0, -1], 'i4')
                    player_one_board[new_pos1[0]][new_pos1[1]] = 2
                    pos1_array[pos1_end_pointer] = new_pos1
                    pos1_end_pointer += 1

        player_two_board = np.zeros((size[0], size[1]), 'i4')
        start_index = pos2_start_pointer
        pos2_start_pointer = pos2_end_pointer
        for index in range(start_index, pos2_end_pointer):
            pos2 = pos2_array[index]
            if pos2[0] + 1 != size[0]:
                if board[pos2[0] + 1][pos2[1]] == 0 and player_two_board[pos2[0] + 1][pos2[1]] == 0:
                    new_pos2 = pos2 + np.array([1, 0], 'i4')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[1] + 1 != size[1]:
                if board[pos2[0]][pos2[1] + 1] == 0 and player_two_board[pos2[0]][pos2[1] + 1] == 0:
                    new_pos2 = pos2 + np.array([0, 1], 'i4')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[0] != 0:
                if board[pos2[0] - 1][pos2[1]] == 0 and player_two_board[pos2[0] - 1][pos2[1]] == 0:
                    new_pos2 = pos2 + np.array([-1, 0], 'i4')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
            if pos2[1] != 0:
                if board[pos2[0]][pos2[1] - 1] == 0 and player_two_board[pos2[0]][pos2[1] - 1] == 0:
                    new_pos2 = pos2 + np.array([0, -1], 'i4')
                    player_two_board[new_pos2[0]][new_pos2[1]] = 3
                    pos2_array[pos2_end_pointer] = new_pos2
                    pos2_end_pointer += 1
        board += player_one_board + player_two_board
    return pos1_end_pointer - pos2_end_pointer

@njit()
def one_player_flood_fill_eval(board, snake_pos, size):
    board = np.copy(board)
    board = np.clip((board[0] - 1) * 4 + board[1], 0, 1000)
    find_tail, count = flood_fill(board, snake_pos, False, 0, size)
    if find_tail:
        return count / 4 + 0.0
    return -50 + count / 4 + 0.0

@njit()
def two_player_flood_fill_eval(board, snake_pos, size):
    board = np.copy(board)
    board = np.clip((board[0] - 1) * 4 + board[1] + board[2], 0, 1000)
    find_tail, count = flood_fill(board, snake_pos, False, 0, size)
    if find_tail:
        return count / 4 + 0.0
    return -50 + count / 4 + 0.0


@njit()
def two_point_flood_fill_eval(board, snakes_pos, size):
    board = np.copy(board)
    board = np.clip((board[0] - 1) * 4 + board[1] + board[2], 0, 1000)
    count = two_point_flood_fill(board, snakes_pos[0], snakes_pos[1], size)
    return count / 4 + 0.0

@njit()
def one_tree_low_depth(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, score, move_number, size, player):
    score += 0.00001
    best_move = 4
    board = np.copy(board)
    snakes_pos = np.copy(snakes_pos)
    wall_spawn = np.copy(wall_spawn)
    snakes_lenght = np.copy(snakes_lenght)
    snakes_alive = np.copy(snakes_alive)
    if np.sum(snakes_alive) == 0:
        return snakes_lenght[0] + score + move_number - 1000, 4
    elif depth <= 0:
        return snakes_lenght[0] + score + one_player_flood_fill_eval(board, snakes_pos[0], size), 4
    else:
        maxeval = -20000
        start_move = np.random.randint(0, 4)
        move_list = np.array([(start_move + i) % 4 for i in range(4)])
        for move in move_list:
            moves = np.array([move])
            new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
            if np.sum(new_snakes_lenght) == np.sum(snakes_lenght):
                tree_eval, next_move = one_tree_low_depth(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score, move_number, size, player)
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
            else:
                lenght = 4
                tree_eval_list = np.zeros(lenght)
                for i in range(lenght):
                    new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
                    tree_eval_list[i], next_move = one_tree_low_depth(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score + depth / 5000, move_number, size, player)
                tree_eval = sum(tree_eval_list) / lenght
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
    return maxeval, best_move

@njit
def two_player_tree_search(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, score, move_number, size, player):
    score += 0.00001
    best_move = 4
    board = np.copy(board)
    snakes_pos = np.copy(snakes_pos)
    wall_spawn = np.copy(wall_spawn)
    snakes_lenght = np.copy(snakes_lenght)
    snakes_alive = np.copy(snakes_alive)
    if not snakes_alive[player]:
        return snakes_lenght[player] + score + move_number - 1000, 4
    elif depth <= 0:
        return snakes_lenght[player] + score + two_player_flood_fill_eval(board, snakes_pos[player], size), 4
    else:
        maxeval = -20000
        start_move = np.random.randint(0, 4)
        move_list = np.array([(start_move + i) % 4 for i in range(4)])
        for move in move_list:
            moves = np.array([4, 4])
            moves[player] = move
            new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
            if np.sum(new_snakes_lenght) <= np.sum(snakes_lenght):
                tree_eval, next_move = two_player_tree_search(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score, move_number, size, player)
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
            else:
                lenght = 4
                tree_eval_list = np.zeros(lenght)
                for i in range(lenght):
                    new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
                    tree_eval_list[i], next_move = two_player_tree_search(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score + depth / 500, move_number, size, player)
                tree_eval = sum(tree_eval_list) / lenght
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
    return maxeval, best_move

@njit()
def two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, score, move_number, size, player, moves, alpha, beta, model):
    maxeval = 0.0
    score += 1
    best_moves = np.array([4, 4], 'i4')
    board = np.copy(board)
    snakes_pos = np.copy(snakes_pos)
    wall_spawn = np.copy(wall_spawn)
    snakes_lenght = np.copy(snakes_lenght)
    snakes_alive = np.copy(snakes_alive)
    if depth % 2 == 0 and moves[0] != 4:
        board, snakes_pos, snakes_lenght, snakes_alive, wall_spawn = move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
        moves = np.array([4, 4], 'i4')
    if np.sum(snakes_alive) <= 1:
        if np.sum(snakes_alive) == 0:
            return 0.0, np.array([4, 4], 'i4')
        if not snakes_alive[1]:
            return 10000.0 - score, np.array([4, 4], 'i4')
        if not snakes_alive[0]:
            return -10000.0 + score, np.array([4, 4], 'i4')
    elif depth <= 0:
        if model == 0:
            flood_fill_eval = two_player_flood_fill_eval(board, snakes_pos[0], size) - two_player_flood_fill_eval(board, snakes_pos[1], size)
        elif model == 1:
            flood_fill_eval = two_point_flood_fill_eval(board, snakes_pos, size)
        return snakes_lenght[0] - snakes_lenght[1] + flood_fill_eval, np.array([4, 4], 'i4')
    else:
        if player == 0:
            maxeval = -20000.0
            start_move = np.random.randint(0, 4)
            move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i4')
            for move in move_list:
                moves[player] = move
                tree_eval, next_move = two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth - 1, score, move_number, size, (player + 1) % 2, moves, alpha, beta, model)
                if tree_eval > maxeval:
                    best_moves = np.array([move, next_move[1]], 'i4')
                    maxeval = tree_eval
                alpha = max(alpha, tree_eval)
                if beta <= alpha:
                    break
        else:
            maxeval = 20000.0
            start_move = np.random.randint(0, 4)
            move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i4')
            for move in move_list:
                moves[player] = move
                tree_eval, next_move = two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth - 1, score, move_number, size, (player + 1) % 2, moves, alpha, beta, model)
                if tree_eval < maxeval:
                    best_moves = np.array([next_move[0], move], 'i4')
                    maxeval = tree_eval
                beta = min(beta, tree_eval)
                if beta <= alpha:
                    break
    return maxeval, best_moves

def ai_player_one(input_board, depth, player):
    board = np.copy(input_board.board)
    snakes_pos = np.copy(input_board.snakes_pos)
    wall_spawn = np.copy(input_board.wall_spawn)
    snakes_lenght = np.copy(input_board.snakes_lenght)
    snakes_alive = np.copy(input_board.snakes_alive)
    return two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, 0.0, input_board.move_count, input_board.size, 1, np.array([4, 4], 'i4'), -10000.0, 10000.0, 1)

def ai_player_two(input_board, depth, player):
    board = np.copy(input_board.board)
    snakes_pos = np.copy(input_board.snakes_pos)
    wall_spawn = np.copy(input_board.wall_spawn)
    snakes_lenght = np.copy(input_board.snakes_lenght)
    snakes_alive = np.copy(input_board.snakes_alive)
    return two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, 0.0, input_board.move_count, input_board.size, 1, np.array([4, 4], 'i4'), -10000.0, 10000.0, 0)

# two_tree_low_depth(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, 0.0, input_board.move_count, input_board.size, player)

def eval_board(board, player, depth, ai_model):
    match ai_model:
        case 1:
            print(ai_model)
        case 16:
            print(ai_model)
        case 17:
            return two_player_tree_search(board.board, board.snakes_pos, board.snakes_lenght, board.wall_spawn, board.snakes_alive, depth, 0.0, board.move_count, board.size, player)
        case 18:
            return two_player_minmax(board.board, board.snakes_pos, board.snakes_lenght, board.wall_spawn, board.snakes_alive, depth * 2, 0.0, board.move_count, board.size, 0, np.array([4, 4], 'i4'), -10000.0, 10000.0, 0)
        case 19:
            return two_player_minmax(board.board, board.snakes_pos, board.snakes_lenght, board.wall_spawn, board.snakes_alive, depth * 2, 0.0, board.move_count, board.size, 0, np.array([4, 4], 'i4'), -10000.0, 10000.0, 1)
    print('no')

def get_move_in_time(board, run_time, player, ai_model):
    if board.number_of_players == 1:
        ai_model = 1
    elif board.number_of_players == 2:
        ai_model = ai_model + 16
    total_time = time.time()
    total_move_time = 0
    depth = 0
    while total_move_time < run_time:
        depth += 1
        depth = round(depth + depth ** 3 / 1000)
        eval, move = eval_board(board, player, depth, ai_model)
        end = time.time()
        total_move_time = (end - total_time)
        if abs(eval) > 200:
            break
        if depth > 100:
            break
    print(depth, eval)
    return move