import numpy as np
import numpy.typing as npt
from game import (
    BLOCK_SIZE, 
    Coord,
    Direction, 
    Game
)

class Agent:
    
    def __init__(self) -> None:
        pass
    
    def state(self, game) -> np.ndarray:
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
        
        danger_left     = int(heading_up and game.is_collision(block_left) or 
                              heading_down and game.is_collision(block_right) or 
                              heading_left and game.is_collision(block_down) or 
                              heading_right and game.is_collision(block_up))
            
        danger_right    = int(heading_up and game.is_collision(block_right) or 
                              heading_down and game.is_collision(block_left) or 
                              heading_left and game.is_collision(block_up) or 
                              heading_right and game.is_collision(block_down))
        
        food_up    = int(game.food.y < head.y)
        food_down  = int(game.food.y > head.y)
        food_left  = int(game.food.x < head.x)
        food_right = int(game.food.x > head.x)
        state: np.ndarray = np.array([danger_straight, 
                                        danger_left, 
                                        danger_right, 
                                        food_up, 
                                        food_down, 
                                        food_left, 
                                        food_right
                                        ], dtype=int)
        return state
    
if __name__ == "__main__":
    agent = Agent()
    game = Game(50,50)
    agent.state(game)
