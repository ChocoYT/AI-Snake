import torch
import random
import numpy as np
from collections import deque
from snake_env import SnakeEnv, Direction, Point, BLOCK_SIZE
from model import LinearQnet, QTrainer
import os

from helper import plot
from threading import Thread

MAX_MEM = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.gameNum = 0
        self.epsilon = 0  # random
        self.gamma = 0.9  # discount
        self.memory = deque(maxlen=MAX_MEM)
        self.model = LinearQnet(11, 512, 3)
        
        if os.path.exists('./models/model1.pth'):
            checkpoint = torch.load('./models/model1.pth')
            self.model.load_state_dict(checkpoint)
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def getState(game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.isCollision(point_r)) or
            (dir_l and game.isCollision(point_l)) or
            (dir_u and game.isCollision(point_u)) or
            (dir_d and game.isCollision(point_d)),

            # Danger Right
            (dir_u and game.isCollision(point_r)) or
            (dir_d and game.isCollision(point_l)) or
            (dir_l and game.isCollision(point_u)) or
            (dir_r and game.isCollision(point_d)),

            # Danger Left
            (dir_d and game.isCollision(point_r)) or
            (dir_u and game.isCollision(point_l)) or
            (dir_r and game.isCollision(point_u)) or
            (dir_l and game.isCollision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food Left
            game.food.x > game.head.x,  # Food Right
            game.food.y < game.head.y,  # Food Up
            game.food.y > game.head.y,  # Food Down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainLongMem(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def trainShortMem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def getAction(self, state):
        self.epsilon = 80 - self.gameNum
        nextMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            nextMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            nextMove[move] = 1

        return nextMove


def getNewThread(func, *args):
    thread = Thread(target=func, args=args)
    thread.start()
    thread.join()


def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    
    record = 0
    
    agent = Agent()
    game = SnakeEnv()
    
    while True:
        # Get Old State
        oldState = agent.getState(game)
        
        # Get Move
        nextMove = agent.getAction(oldState)

        # Perform move & Get New State
        reward, done, score = game.playStep(nextMove)
        newState = agent.getState(game)

        # Train Short Memory
        agent.trainShortMem(oldState, nextMove, reward, newState, done)

        agent.remember(oldState, nextMove, reward, newState, done)

        if done:
            # Train and Plot Results
            game.reset()
            agent.gameNum += 1
            agent.trainLongMem()

            if score > record:
                record = score
                agent.model.save()
                
            totalScore += score

            #plotScores.append(score)
            #plotMeanScores.append(totalScore / agent.gameNum)
            #plot(plotScores, plotMeanScores)
            #getNewThread(plot, plotScores, plotMeanScores)


if __name__ == '__main__':
    train()
