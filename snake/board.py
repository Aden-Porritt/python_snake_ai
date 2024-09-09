import pygame
import numpy as np
import snake.snake_eng as eng

class Board():
    def __init__(self, size, number_of_players, WIN, SQRT_SIZE) -> None:
        self.number_of_players = number_of_players
        self.board, self.snakes_pos, self.snakes_lenght, self.wall_spawn, self.snakes_alive = eng.make_board(size, number_of_players)
        self.size = size
        self.move_count = 0
        self.snakes_head_colour = np.array([[29, 20, 217], [211, 224, 18]])
        self.snakes_tail_colour = np.array([[224, 28, 18], [18, 224, 193]])
        self.WIN = WIN
        self.SQRT_SIZE = SQRT_SIZE

    def move_snake(self, move):
        self.board, self.snakes_pos, self.snakes_lenght, self.snakes_alive, self.wall_spawn = eng.move_snake(move, self.board, self.snakes_pos, self.snakes_lenght, self.wall_spawn, self.snakes_alive, self.size)
        return self.snakes_alive
    
    def set_board(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.board[0][row][col] == self.snake_lenght:
                    self.snake_pos = np.array([row, col])
                if self.board[2][row][col] == 1:
                    self.board, self.wall_spawn = eng.change_wall_spawn(self.board, np.array([row, col]), self.wall_spawn, self.size)
    
    def draw_board(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                board_color = (21, 237, 72) if (row + col) % 2 == 0 else (28, 201, 69)

                pygame.draw.rect(self.WIN, board_color, (col * self.SQRT_SIZE, row * self.SQRT_SIZE, self.SQRT_SIZE, self.SQRT_SIZE))
                
                if self.board[0][row][col] == 2:
                    pygame.draw.rect(self.WIN, (1, 1, 1), (col * self.SQRT_SIZE, row * self.SQRT_SIZE, self.SQRT_SIZE, self.SQRT_SIZE))

                elif self.board[0][row][col] == 1:
                    pygame.draw.circle(self.WIN, (201, 28, 51), ((col + 1/2) * self.SQRT_SIZE, (row + 1/2) * self.SQRT_SIZE), self.SQRT_SIZE / 2)

                elif np.sum(self.board[:, row, col]) != 0:
                    for index in range(self.number_of_players):
                        if self.board[1 + index][row][col] > 0:
                            change_in_snake_colour = (self.snakes_head_colour[index] - self.snakes_tail_colour[index]) / self.snakes_lenght[index]
                            snake_color = self.snakes_tail_colour[index] + change_in_snake_colour * self.board[1 + index][row][col]
                            pygame.draw.circle(self.WIN, snake_color, ((col + 1/2) * self.SQRT_SIZE, (row + 1/2) * self.SQRT_SIZE), 3 * self.SQRT_SIZE / 8)
                            if row + 1 != self.size[0]:
                                if (self.board[1 + index][row + 1][col] == self.board[1 + index][row][col] - 1 or self.board[1 + index][row + 1][col] == self.board[1 + index][row][col] + 1) and self.board[1 + index][row + 1][col] != 0:
                                    pygame.draw.rect(self.WIN, snake_color, ((col + 1/8) * self.SQRT_SIZE, (row + 1/2) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE * 1/2))
                            if row - 1 != -1:
                                if (self.board[1 + index][row - 1][col] == self.board[1 + index][row][col] - 1 or self.board[1 + index][row - 1][col] == self.board[1 + index][row][col] + 1) and self.board[1 + index][row - 1][col] != 0:
                                    pygame.draw.rect(self.WIN, snake_color, ((col + 1/8) * self.SQRT_SIZE, (row) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE * 1/2))
                            if col + 1 != self.size[1]:
                                if (self.board[1 + index][row][col + 1] == self.board[1 + index][row][col] - 1 or self.board[1 + index][row][col + 1] == self.board[1 + index][row][col] + 1) and self.board[1 + index][row][col + 1] != 0:
                                    pygame.draw.rect(self.WIN, snake_color, ((col + 1/2) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/2, self.SQRT_SIZE * 3/4))
                            if col - 1 != -1:
                                if (self.board[1 + index][row][col - 1] == self.board[1 + index][row][col] - 1 or self.board[1 + index][row][col - 1] == self.board[1 + index][row][col] + 1) and self.board[1 + index][row][col - 1] != 0:
                                    pygame.draw.rect(self.WIN, snake_color, ((col) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/2, self.SQRT_SIZE * 3/4))

                            if row + 1 != self.size[0]:
                                if self.board[1 + index][row + 1][col] == self.board[0][row][col] - 1 and self.board[1 + index][row + 1][col] != 0:
                                    new_snake_color = snake_color - change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 1/8) * self.SQRT_SIZE, (row + 7/8) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE / 8 + 1))

                                if self.board[1 + index][row + 1][col] == self.board[1 + index][row][col] + 1 and self.board[1 + index][row + 1][col] != 0:
                                    new_snake_color = snake_color + change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 1/8) * self.SQRT_SIZE, (row + 7/8) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE / 8 + 1))
                            
                            if row - 1 != -1:
                                if self.board[1 + index][row - 1][col] == self.board[1 + index][row][col] - 1 and self.board[1 + index][row - 1][col] != 0:
                                    new_snake_color = snake_color - change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 1/8) * self.SQRT_SIZE, (row) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE / 8 + 1))

                                if self.board[1 + index][row - 1][col] == self.board[1 + index][row][col] + 1 and self.board[1 + index][row - 1][col] != 0:
                                    new_snake_color = snake_color + change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 1/8) * self.SQRT_SIZE, (row) * self.SQRT_SIZE, self.SQRT_SIZE * 3/4, self.SQRT_SIZE / 8 + 1))
                            
                            if col + 1 != self.size[1]:
                                if self.board[1 + index][row][col + 1] == self.board[1 + index][row][col] - 1 and self.board[1 + index][row][col + 1] != 0:
                                    new_snake_color = snake_color - change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 7/8) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/8 + 1, self.SQRT_SIZE * 3/4))

                                if self.board[1 + index][row][col + 1] == self.board[1 + index][row][col] + 1 and self.board[1 + index][row][col + 1] != 0:
                                    new_snake_color = snake_color + change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col + 7/8) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/8 + 1, self.SQRT_SIZE * 3/4))

                            if col - 1 != -1:
                                if self.board[1 + index][row][col - 1] == self.board[1 + index][row][col] - 1 and self.board[1 + index][row][col - 1] != 0:
                                    new_snake_color = snake_color - change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/8 + 1, self.SQRT_SIZE * 3/4))

                                if self.board[1 + index][row][col - 1] == self.board[1 + index][row][col] + 1 and self.board[1 + index][row][col - 1] != 0:
                                    new_snake_color = snake_color + change_in_snake_colour / 3
                                    pygame.draw.rect(self.WIN, new_snake_color, ((col) * self.SQRT_SIZE, (row + 1/8) * self.SQRT_SIZE, self.SQRT_SIZE * 1/8 + 1, self.SQRT_SIZE * 3/4))
                
                for player in range(self.number_of_players):
                    pygame.draw.circle(self.WIN, (255, 255, 255), ((self.snakes_pos[player][1] + 1/2) * self.SQRT_SIZE, (self.snakes_pos[player][0] + 1/2) * self.SQRT_SIZE), 3 * self.SQRT_SIZE / 12)
                    pygame.draw.circle(self.WIN, (1, 1, 1), ((self.snakes_pos[player][1] + 1/2) * self.SQRT_SIZE, (self.snakes_pos[player][0] + 1/2) * self.SQRT_SIZE), self.SQRT_SIZE / 10)
