import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from sys import exit

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", 'x, y')

BLOCK_SIZE = 10
SPEED = 9999

class SnakeEnv:
    def __init__(self, width=640, height=480):
        self.width= width
        self.height = height
        self.frameIter = None
        self.food = None
        self.score = None
        self.head = None
        self.snake = None
        
        # Init Display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.direction = None
        
        # Init Game State
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.food = None
        self._placeFood()
        self.frameIter = 0

    def _placeFood(self):
        x = random.randint(0, (self.width  - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._placeFood()

    def playStep(self, action):
        self.frameIter += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
                
        # Move Snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check if Game Over
        reward = 0
        gameOver = False
        if self.isCollision() or self.frameIter > (len(self.snake) * 100):
            gameOver = True
            reward = -10
            return reward, gameOver, self.score

        # Check if Food Eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._placeFood()
        else:
            self.snake.pop()

        # Refresh Display
        self._updateUi()
        dt = self.clock.tick(SPEED)
        pygame.display.set_caption(f"Snake Game - FPS: {dt}")
        
        return reward, gameOver, self.score

    def isCollision(self, pt=None):
        # Hits Boundary
        if pt is None:
            pt = self.head
        
        if (pt.x > self.width - BLOCK_SIZE) or (pt.y > self.height - BLOCK_SIZE) or (pt.x < 0) or (pt.y < 0):
            return True
        # Hits Itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [Straight, Right, Left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _updateUi(self):
        self.display.fill((0, 0, 0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Score is : {self.score}", True, (255, 255, 255))
        self.display.blit(text, [3, 0])
        pygame.display.flip()


if __name__ == '__main__':
    exit()
