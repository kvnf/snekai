from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from collections import deque
from game import (
    BLOCK_SIZE,
    STRAIGHT,
    LEFT,
    RIGHT,

    Game,
)
from IPython import display
from math import exp
from model import DQN, Trainer
from random import choice, sample, randint, random
from typing import (
    Deque,
    List,
    Tuple,
)
from custom_types import (
    Action,
    Coord,
    Direction,
    MemoryCell,
)


class PlotHandler:
    def __init__(self, loss: List[float] = [],
                 record: int = 0,
                 scores: List[int] = [],
                 predicted_ratio: List[float] = []) -> None:
        self.scores = scores
        self.records: List[int] = [record]
        self.record_games: List[int] = [0]
        self.predicted_ratio = predicted_ratio
        self.total_score = 0

        self.loss = loss
        self.max_loss = np.finfo(float).tiny

        self.fig, self.axs = plt.subplots(2, 1)
        self.ax2 = self.axs[0].twinx()
        plt.ion()

        self.ma: List[float] = []
        self.mean: List[float] = []
        self.mean_diff: List[float] = []

    def moving_avg(self, x: List, n: int) -> None:
        self.ma.append(sum(x[-n:]) / n)

    def _scores_calc(self, avg: int, games: int,  loss: float, record: int, score: int, predicted_ratio: float) -> None:
        # if loss > self.max_loss:
        #     self.max_loss = loss
        # self.axs[1].plot(self.loss, color='red')
        # self.axs[1].set_ylabel("Loss")
        if record > self.records[-1]:
            self.records.append(record)
            self.record_games.append(games - 1)
        self.total_score += score
        self.scores.append(score)
        self.predicted_ratio.append(predicted_ratio)
        self.mean.append(self.total_score / games)
        self.moving_avg(self.scores, avg)
        self.mean_diff.append(self.ma[-1] - self.mean[-1])
        self.loss.append(loss)

    def plot(self, avg: int, games: int,  loss: float, record: int, score: int, predicted_ratio: float) -> None:
        self._scores_calc(avg, games,  loss, record, score, predicted_ratio)
        self.axs[0].plot(self.scores, color='red')
        self.axs[0].scatter(self.record_games, self.records, color='red')
        self.axs[0].plot(self.mean, color='green')
        self.axs[0].plot(self.ma, color='blue')
        self.axs[0].set_ylabel("Game Score")
        self.axs[0].legend(
            ['Score', f'Record: {record}', 'Score mean',
                f'Score avg. {avg} games'],
            loc='upper left')
        self.ax2.plot(self.predicted_ratio, ls='--', color='orange')
        self.ax2.set_ylabel('steps predicted / total steps', color='orange')
        self.axs[1].plot(self.mean_diff, color='red')
        self.axs[1].set_ylabel(f'Avg {avg} games - Score mean')
        self.axs[1].set_xlabel("Games Played")
        self.fig.show()
        plt.pause(1)


MEM_SIZE: int = 100000
BATCH_SIZE: int = 1000
EPS_START: float = 1.0
EPS_END: float = -0.3
LEARNING_RATE: float = 0.0001
GAMMA: float = 0.9
# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, decay=100) -> None:
        self.memory: Deque[MemoryCell] = deque(maxlen=MEM_SIZE)
        self.eps_decay = decay
        # in_size == len(self.state)
        # out_size == len(STRAIGHT, LEFT,RIGHT)
        self.dqn = DQN(in_size=11, inner_size=256, out_size=3).to(device)
        self.trainer = Trainer(self.dqn, LEARNING_RATE, GAMMA)
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
        if len(self.memory) < BATCH_SIZE:
            train_set = list(self.memory)
        else:
            train_set = list(sample(self.memory, BATCH_SIZE))

        for el in train_set:
            loss = self.trainer.train(el)
        return loss

    def select_action(self, state: np.ndarray) -> Action:
        # model decides action based on epsilon greedy algorithm
        actions: Tuple[Action, Action, Action] = (STRAIGHT, RIGHT, LEFT)
        action: Action = STRAIGHT
        if random() * EPS_START > self._epsilon_threshold():
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
        eps_threshold: float = EPS_END + \
            (EPS_START - EPS_END) * exp(-self.games_played/self.eps_decay)
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
