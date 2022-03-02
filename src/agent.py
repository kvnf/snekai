import numpy as np
import torch
from collections import deque
from .game import (
    BLOCK_SIZE,
    STRAIGHT,
    LEFT,
    RIGHT,

    Game,
)
from math import exp
from .model import DQN, Trainer
from random import choice, sample, randint, random
from typing import (
    Deque,
    List,
    Tuple,
)
from .custom_types import (
    Action,
    Coord,
    Direction,
    MemoryCell,
)
from .plot_handler import PlotHandler

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, mem_size: int = 100000,
                 batch_size: int = 1000,
                 eps_start: float = 1.0,
                 eps_end: float = -0.3,
                 learning_rate=0.0001,
                 gamma: float = 0.9,
                 decay: int = 100) -> None:
        self.memory: Deque[MemoryCell] = deque(maxlen=mem_size)
        self.eps_decay = decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.batch_size = batch_size
        # in_size == len(self.state)
        # out_size == len(STRAIGHT, LEFT,RIGHT)

       # model
        self.dqn = DQN(in_size=11, inner_size=256, out_size=3).to(device)
        self.trainer = Trainer(self.dqn, learning_rate, gamma)

        self.games_played: int = 0
        self.step_predictions = 0
        self.step_random = 0

    def state(self, game: Game) -> np.ndarray:
        head:        Coord = game.head
        block_up:    Coord = Coord(head.x, head.y-BLOCK_SIZE)
        block_down:  Coord = Coord(head.x, head.y+BLOCK_SIZE)
        block_left:  Coord = Coord(head.x-BLOCK_SIZE, head.y)
        block_right: Coord = Coord(head.x+BLOCK_SIZE, head.y)

        heading_up:    bool = game.direction == Direction.UP
        heading_down:  bool = game.direction == Direction.DOWN
        heading_left:  bool = game.direction == Direction.LEFT
        heading_right: bool = game.direction == Direction.RIGHT

        danger_straight = int(heading_up and game.is_collision(block_up) or
                              heading_down and game.is_collision(block_down) or
                              heading_left and game.is_collision(block_left) or
                              heading_right and game.is_collision(block_right))

        danger_left = int(heading_up and game.is_collision(block_left) or
                          heading_down and game.is_collision(block_right) or
                          heading_left and game.is_collision(block_down) or
                          heading_right and game.is_collision(block_up))

        danger_right = int(heading_up and game.is_collision(block_right) or
                           heading_down and game.is_collision(block_left) or
                           heading_left and game.is_collision(block_up) or
                           heading_right and game.is_collision(block_down))

        food_up = int(game.food.y < head.y)
        food_down = int(game.food.y > head.y)
        food_left = int(game.food.x < head.x)
        food_right = int(game.food.x > head.x)

        state: np.ndarray = np.array([danger_straight,
                                      danger_left,
                                      danger_right,
                                      food_up,
                                      food_down,
                                      food_left,
                                      food_right,
                                      heading_up,
                                      heading_down,
                                      heading_left,
                                      heading_right,
                                      ], dtype=int)
        return state

    def enqueue_memory(self, mem_cell: MemoryCell) -> None:
        self.memory.append(mem_cell)
        pass

    def step_train(self, mem_cell: MemoryCell) -> None:
        self.trainer.train(mem_cell)

    def train(self) -> float:
        loss: float = 0
        if len(self.memory) < self.batch_size:
            train_set = list(self.memory)
        else:
            train_set = list(sample(self.memory, self.batch_size))

        for el in train_set:
            loss = self.trainer.train(el)
        return loss

    def select_action(self, state: np.ndarray) -> Action:
        # model decides action based on epsilon greedy algorithm
        actions: Tuple[Action, Action, Action] = (STRAIGHT, RIGHT, LEFT)
        action: Action = STRAIGHT
        if random() * self.eps_start > self._epsilon_threshold():
            # call model & predict
            state_tensor = torch.tensor(
                state, device=device, dtype=torch.float)
            prediction = self.dqn(state_tensor)
            idx = torch.argmax(prediction).tolist()  # = 0,1,2
            action = actions[idx]
            self.step_predictions += 1
        else:
            # random action
            action = choice(actions)
            self.step_random += 1

        return action

    def _epsilon_threshold(self) -> float:
        eps_threshold: float = self.eps_end + \
            (self.eps_start - self.eps_end) * \
            exp(-self.games_played/self.eps_decay)
        return eps_threshold


def train_agent(agent: Agent, game: Game) -> None:
    record: int = 0
    loss: float = 0
    plot = PlotHandler()
    while True:
        # get state before action
        curr_state: np.ndarray = agent.state(game)
        # get and execute move based on current game state
        action: Action = agent.select_action(curr_state)
        reward, game_score, game_over = game.game_step(action)
        new_state: np.ndarray = agent.state(game)
        step_memory = MemoryCell(curr_state, new_state, action,
                                 game_over, reward)
        # train
        agent.step_train(step_memory)
        agent.enqueue_memory(step_memory)

        if game_over:
            if game_score > record:
                record = game_score
            game.restart()
            agent.games_played += 1
            loss = agent.train()
            predicted_ratio = agent.step_predictions / \
                (agent.step_predictions + agent.step_random)
            # plot
            plot.plot(20, agent.games_played, loss,
                      record, game_score, predicted_ratio)
            agent.step_predictions = 0
            agent.step_random = 0


if __name__ == "__main__":

    agent = Agent()
    game = Game(640, 480)
    agent.state(game)
    train_agent(agent, game)
