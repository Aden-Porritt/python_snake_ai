import numpy as np
import snake
import snake.board as snake_board
import snake.snake_eng as eng

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

def main():
    win_draw_loss = np.zeros(3, 'i4')
    number_of_players = 2
    clock = pygame.time.Clock()
    moves = np.array([[4 for _ in range(10)] for i in range(number_of_players)])
    moves_end_pointer = np.array([0 for i in range(number_of_players)])
    run = True
    while run:
        moves = np.array([[4 for _ in range(10)] for i in range(number_of_players)])
        moves_end_pointer = np.array([0 for i in range(number_of_players)])
        move_number = 0
        board = snake_board.Board(SIZE, number_of_players, WIN, SQRT_SIZE)
        board.draw_board()
        pygame.display.flip()
        while np.sum(board.snakes_alive) > 1:
            next_move = False
            clock.tick(7.5)
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
            ai_move_one = eng.get_move_in_time(board, 0.01, 0, 2)
            ai_move_two = eng.get_move_in_time(board, 0.01, 1, 3)
            print(ai_move_one)
            # ai_move = np.random.randint(0, 4)
            # for move in moves:
            #     if move[0][0] == 4:
            #         break
            if True: #moves[0][0] != 4:
                if move_number % 1 == 0:
                    board.move_snake(np.array([ai_move_one[0], ai_move_two[1]], 'i4'))
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

SIZE = np.array([11, 12])

if __name__ == '__main__':
    import pygame
    from pygame.locals import *
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg
    import pylab
    pygame.init()
    board = snake_board.Board(SIZE, 2, 1, 1)
    ai_move = eng.get_move_in_time(board, 0.005, 0, 2)
    board.move_snake(np.array([0]))
    board = snake_board.Board(SIZE, 2, 1, 1)
    ai_move = eng.get_move_in_time(board, 0.005, 0, 3)
    print('start')
    HEIGHT, WIDTH, SQRT_SIZE = get_win_size(SIZE)
    print(HEIGHT, WIDTH)
    WIDTH += 400
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("game")
    pylab.figure(figsize=[4, 4], dpi=100)
    main()
    print('f')