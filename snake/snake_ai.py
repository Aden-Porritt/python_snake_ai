import numpy as np
import time
from numba import njit
import requests
import json
import snake.snake_eng as eng

@njit()
def flood_fill(board, snake_pos, find_tail, count, size):
    snake_pos = np.copy(snake_pos)
    if snake_pos[0] + 1 != size[0]:
        if board[snake_pos[0] + 1][snake_pos[1]] == 1:
            find_tail = True
        elif board[snake_pos[0] + 1][snake_pos[1]] == 0:
            new_snake_pos = snake_pos + np.array([1, 0], 'i8')
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[1] + 1 != size[1]:
        if board[snake_pos[0]][snake_pos[1] + 1] == 1:
            find_tail = True
        elif board[snake_pos[0]][snake_pos[1] + 1] == 0:
            new_snake_pos = snake_pos + np.array([0, 1], 'i8')
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[0] != 0:
        if board[snake_pos[0] - 1][snake_pos[1]] == 1:
            find_tail = True
        elif board[snake_pos[0] - 1][snake_pos[1]] == 0:
            new_snake_pos = snake_pos + np.array([-1, 0], 'i8')
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    if snake_pos[1] != 0:
        if board[snake_pos[0]][snake_pos[1] - 1] == 1:
            find_tail = True
        elif board[snake_pos[0]][snake_pos[1] - 1] == 0:
            new_snake_pos = snake_pos + np.array([0, -1], 'i8')
            board[new_snake_pos[0]][new_snake_pos[1]] = 2
            count += 1
            find_tail, count = flood_fill(board, new_snake_pos, find_tail, count, size)
    return find_tail, count

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
        move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i8')
        for move in move_list:
            moves = np.array([move], 'i8')
            new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = eng.move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
            if np.sum(new_snakes_lenght) == np.sum(snakes_lenght):
                tree_eval, next_move = one_tree_low_depth(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score, move_number, size, player)
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
            else:
                lenght = 4
                tree_eval_list = np.zeros(lenght)
                for i in range(lenght):
                    new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = eng.move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
                    tree_eval_list[i], next_move = one_tree_low_depth(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score + depth / 5000, move_number, size, player)
                tree_eval = sum(tree_eval_list) / lenght
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
    return maxeval, best_move

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
        move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i8')
        for move in move_list:
            moves = np.array([4, 4], 'i8')
            moves[player] = move
            new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = eng.move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
            if np.sum(new_snakes_lenght) <= np.sum(snakes_lenght):
                tree_eval, next_move = two_player_tree_search(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score, move_number, size, player)
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
            else:
                lenght = 4
                tree_eval_list = np.zeros(lenght)
                for i in range(lenght):
                    new_board, new_snakes_pos, new_snakes_lenght, new_snakes_alive, new_wall_spawn = eng.move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
                    tree_eval_list[i], next_move = two_player_tree_search(new_board, new_snakes_pos, new_snakes_lenght, new_wall_spawn, new_snakes_alive, depth - 1, score + depth / 500, move_number, size, player)
                tree_eval = sum(tree_eval_list) / lenght
                if tree_eval > maxeval:
                    best_move = move
                    maxeval = tree_eval
    return maxeval, best_move

def two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, score, move_number, size, player, moves, alpha, beta, model):
    maxeval = 0.0
    score += 1
    best_moves = np.array([4, 4], 'i8')
    board = np.copy(board)
    snakes_pos = np.copy(snakes_pos)
    wall_spawn = np.copy(wall_spawn)
    snakes_lenght = np.copy(snakes_lenght)
    snakes_alive = np.copy(snakes_alive)
    if depth % 2 == 0 and moves[0] != 4:
        board, snakes_pos, snakes_lenght, snakes_alive, wall_spawn = eng.move_snake(moves, board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, size)
        moves = np.array([4, 4], 'i8')
    if np.sum(snakes_alive) <= 1:
        if np.sum(snakes_alive) == 0:
            return 0.0, np.array([4, 4], 'i8')
        if not snakes_alive[1]:
            return 10000.0 - score, np.array([4, 4], 'i8')
        if not snakes_alive[0]:
            return -10000.0 + score, np.array([4, 4], 'i8')
    elif depth <= 0:
        if model == 0:
            flood_fill_eval = two_player_flood_fill_eval(board, snakes_pos[0], size) - two_player_flood_fill_eval(board, snakes_pos[1], size)
        elif model == 1:
            flood_fill_eval = eng.two_point_flood_fill_eval(board, snakes_pos, size)
        return snakes_lenght[0] - snakes_lenght[1] + flood_fill_eval, np.array([4, 4], 'i8')
    else:
        if player == 0:
            maxeval = -20000.0
            start_move = np.random.randint(0, 4)
            move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i8')
            for move in move_list:
                moves[player] = move
                tree_eval, next_move = two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth - 1, score, move_number, size, (player + 1) % 2, moves, alpha, beta, model)
                if tree_eval > maxeval:
                    best_moves = np.array([move, next_move[1]], 'i8')
                    maxeval = tree_eval
                alpha = max(alpha, tree_eval)
                if beta <= alpha:
                    break
        else:
            maxeval = 20000.0
            start_move = np.random.randint(0, 4)
            move_list = np.array([(start_move + i) % 4 for i in range(4)], 'i8')
            for move in move_list:
                moves[player] = move
                tree_eval, next_move = two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth - 1, score, move_number, size, (player + 1) % 2, moves, alpha, beta, model)
                if tree_eval < maxeval:
                    best_moves = np.array([next_move[0], move], 'i8')
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
    return two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, 0.0, input_board.move_count, input_board.size, 1, np.array([4, 4], 'i8'), -10000.0, 10000.0, 1)

def ai_player_two(input_board, depth, player):
    board = np.copy(input_board.board)
    snakes_pos = np.copy(input_board.snakes_pos)
    wall_spawn = np.copy(input_board.wall_spawn)
    snakes_lenght = np.copy(input_board.snakes_lenght)
    snakes_alive = np.copy(input_board.snakes_alive)
    return two_player_minmax(board, snakes_pos, snakes_lenght, wall_spawn, snakes_alive, depth, 0.0, input_board.move_count, input_board.size, 1, np.array([4, 4], 'i8'), -10000.0, 10000.0, 0)

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
            return two_player_minmax(board.board, board.snakes_pos, board.snakes_lenght, board.wall_spawn, board.snakes_alive, depth * 2, 0.0, board.move_count, board.size, 0, np.array([4, 4], 'i8'), -10000.0, 10000.0, 0)
        case 19:
            return two_player_minmax(board.board, board.snakes_pos, board.snakes_lenght, board.wall_spawn, board.snakes_alive, depth * 2, 0.0, board.move_count, board.size, 0, np.array([4, 4], 'i8'), -10000.0, 10000.0, 1)
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

def get_board(test):
    test = json.loads(test)
    width = test['width']
    height = test['height']

    size = np.array([height, width], 'i8')
    board = np.zeros((3, size[0], size[1]), 'i8')

    for index, cell in enumerate(test['cells']):
        y = height - 1 - (index // width)
        x = index % width
        match cell:
            case 'Empty':
                pass
            case 'Wall':
                board[0][y][x] = 2
            case _:
                if 'Apple' in cell:
                    board[0][y][x] = 1
                elif 'Snake' in cell:
                    board[cell['Snake']['id'] + 1][y][x] = cell['Snake']['part'] + 1
                else:
                    print('fuck')
                    a = 3 + ''
    return board, size

def get_str_move(move):
    match move:
        case 0:
            return 'Down'
        case 1:
            return 'Right'
        case 2:
            return 'Up'
        case 3:
            return 'Left'