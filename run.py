import numpy as np
import snake
import requests
import json
import asyncio
import websockets
import tracemalloc

def get_win_size(size):
    max_size = np.array([1600, 800])
    for i in range(100):
        if size[0] * i <= max_size[1] and size[1] * i <= max_size[0]:
            HEIGHT, WIDTH = size[0] * i, size[1] * i
        else: break
    return HEIGHT, WIDTH, i - 1

def display_score(score):
    font_ = pygame.font.Font('freesansbold.ttf', 30)
    str_score = str(score[0]) + " " + str(score[1]) + " " + str(score[2])
    text = font_.render(str_score, True, (255, 255, 255))
    WIN.blit(text, (WIDTH // 2, 20))

        
async def online_game_loop():
    # uri = "ws://bink.eu.org:1234/ws"
    uri = "ws://127.0.0.1:1234/ws"

    async with websockets.connect(uri) as websocket:
        move_count = 0
        while True:
            print('t')
            new_board = await websocket.recv()
            new_board, size = snake.snake_ai.get_board(new_board)
            board = snake.board.Board(size, 2, WIN, SQRT_SIZE)
            board.board = new_board
            board.set_board()
            if np.sum(new_board[0]) == 0:
                move_count = 0
            move = snake.snake_ai.get_move_in_time(board, 0.01, 1, 3)
            move = move[0]
            print(move)
            move = snake.snake_ai.get_str_move(move)
            print(move)
            json_move = json.dumps({'direction': move})
            if True:
                for event in pygame.event.get():
                    pass
                board.draw_board()
                pygame.display.update()
            await websocket.send(json_move)

def main():
    win_draw_loss = np.zeros(3, 'i8')
    number_of_players = 2
    clock = pygame.time.Clock()
    moves = np.array([[4 for _ in range(10)] for i in range(number_of_players)])
    moves_end_pointer = np.array([0 for i in range(number_of_players)])
    run = True
    while run:
        moves = np.array([[4 for _ in range(10)] for i in range(number_of_players)])
        moves_end_pointer = np.array([0 for i in range(number_of_players)])
        move_number = 0
        board = snake.board.Board(SIZE, number_of_players, WIN, SQRT_SIZE)
        board.draw_board()
        pygame.display.flip()
        while np.sum(board.snakes_alive) > 1:
            next_move = False
            clock.tick(200)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    board.snakes_alive = np.array([False for i in range(number_of_players)])
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        board.snakes_alive = np.array([False for i in range(number_of_players)])
                    if moves_end_pointer[0] != 9:
                        if event.key == pygame.K_a:
                            moves[0][moves_end_pointer[0]] = 3
                            moves_end_pointer[0] += 1
                        if event.key == pygame.K_w:
                            moves[0][moves_end_pointer[0]] = 2
                            moves_end_pointer[0] += 1
                        if event.key == pygame.K_d:
                            moves[0][moves_end_pointer[0]] = 1
                            moves_end_pointer[0] += 1
                        if event.key == pygame.K_s:
                            moves[0][moves_end_pointer[0]] = 0
                            moves_end_pointer[0] += 1
                        if event.key == pygame.K_n:
                            next_move = True
            ai_move_one = snake.snake_ai.get_move_in_time(board, 0.01, 0, 3)
            ai_move_two = snake.snake_ai.get_move_in_time(board, 0.01, 0, 3)
            print(ai_move_one)
            # ai_move = np.random.randint(0, 4)
            # for move in moves:
            #     if move[0][0] == 4:
            #         break
            if True: #moves[0][0] != 4:
                if move_number % 1 == 0:
                    board.move_snake(np.array([ai_move_one[0], ai_move_two[1]], 'i8'))
                    moves_end_pointer -= 1
                    moves_end_pointer = np.clip(moves_end_pointer, 0, 10)
                    for index, snake_moves in enumerate(moves):
                        if moves_end_pointer[index] != 0:
                            new_snake_moves = snake_moves[1:]
                            moves[index][-1] = 4
                            for i, next_move in enumerate(new_snake_moves):
                                moves[index][i] = next_move
                move_number += 1
            board.draw_board()
            display_score(win_draw_loss)
            pygame.display.flip()
        if np.sum(board.snakes_alive) == 2:
            pass
        elif np.sum(board.snakes_alive) == 0:
            win_draw_loss[1] += 1
        elif board.snakes_alive[0]:
            win_draw_loss[0] += 1
        elif board.snakes_alive[1]:
            win_draw_loss[2] += 1

SIZE = np.array([9, 10], 'i8')

if __name__ == '__main__':
    import pygame
    from pygame.locals import *
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg
    import pylab
    pygame.init()
    print('start')
    HEIGHT, WIDTH, SQRT_SIZE = get_win_size(SIZE)
    print(HEIGHT, WIDTH)
    WIDTH += 400
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("game")
    pylab.figure(figsize=[4, 4], dpi=100)
    main()
    # tracemalloc.start()
    # asyncio.run(online_game_loop())
    print('f')