import numpy as np
from enum import (
    auto,
    Enum
)

from typing import (
    NamedTuple,
    Tuple
)

Action = Tuple[int, int, int]


class Coord(NamedTuple):
    x: int
    y: int


class Direction(Enum):
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()


class MemoryCell():
    def __init__(self, state_prev: np.ndarray, state_next: np.ndarray, action: Action,
                 game_over: bool, reward: int):
        self.state_prev = state_prev
        self.state_next = state_next
        self.action = action
        self.game_over = game_over
        self.reward = reward
        self.cell: Tuple[np.ndarray, np.ndarray,
                         Action, bool, int] = (state_prev, state_next, action, game_over, reward)
