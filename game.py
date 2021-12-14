import pygame
from enum import (
    auto,
    Enum
)
from random import randint
from typing import (
    List, 
    NamedTuple,
    Optional
)


BLOCK_SIZE: int = 10

BLACK   =   '0x000000'
BLUE    =   '0x0000ff'
GREEN   =   '0x00ff00'
RED     =   '0xff0000'
WHITE   =   '0xffffff'
class Direction(Enum):
    DOWN    =   auto()
    LEFT    =   auto()
    RIGHT   =   auto()    
    UP      =   auto()


class Coord(NamedTuple):
    x  : int
    y  : int


class Game:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.restart()
        
    def game_step(self, action):
        self._move(action)
        self._screen_update()    
    
    def restart(self) -> None:
        # snake stuff
        self.direction: Direction = Direction.RIGHT
        self.head: Coord = Coord(self.width/2, self.height/2)
        self.snake: List[Coord] = [self.head]
        
        # env stuff
        self.score: int = 0
        self.food: Optional[Coord] = None
        self.frame  = 0
        self._spawn_food()
    
              
    
    
    # private
    
    def _move(self, action) -> None:
        pass
    
    def _screen_update(self) -> None:
        self.display.fill(GREEN)
        
        for block in self.snake:
            pygame.draw.rect(self.display, WHITE, 
                             pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))  
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
    
    def _spawn_food(self) -> None:
        x_blocks: int = (self.width - BLOCK_SIZE) // BLOCK_SIZE            
        y_blocks: int = (self.height - BLOCK_SIZE) // BLOCK_SIZE
        
        food_x: int  = randint(0, x_blocks) * BLOCK_SIZE            
        food_y: int  = randint(0, y_blocks) * BLOCK_SIZE
        self.food = Coord(food_x, food_y)          

if __name__ == "__main__":
    game = Game(300, 150)
    def play(game: Game):
        while True:
            game.game_step()

    play(game)