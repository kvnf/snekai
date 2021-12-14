import pygame
from enum import (
    auto,
    Enum
)
from random import (
    choice,
    randint
)
from typing import (
    AsyncIterable,
    List, 
    NamedTuple,
    Optional
)

pygame.init()
font = pygame.font.SysFont('ProggyCleanTTSZ Nerd Font Mono', 30)

BLOCK_SIZE: int = 20
SPEED = 15

BLACK   =   '0x000000'
BLUE    =   '0x0000ff'
GREEN   =   '0x00ff00'
RED     =   '0xff0000'
WHITE   =   '0xffffff'

STRAIGHT =   [1, 0, 0]
RIGHT    =   [0, 1, 0]
LEFT     =   [0, 0, 1]

AI: bool = True
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
        
    def restart(self) -> None:
        # snake stuff
        self.direction: Direction = Direction.RIGHT
        self.head: Coord = Coord(self.width//2, self.height//2)
        self.snake: List[Coord] = [self.head, 
                                   Coord(self.head.x - BLOCK_SIZE, self.head.y)]
        
        # env stuff
        self.score: int = 0
        self.food: Optional[Coord] = None
        self.step  = 0
        self._spawn_food()
    
    def game_step(self, action) -> bool:
        if not AI: 
            self.usr_input()
        self._move(action)
        self.snake.insert(0, self.head)
        game_over = False
        if self.is_collision():
            game_over = True
        if self.head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()
            
        self._screen_update()
        self.clock.tick(SPEED)
        self.step += 1
        return game_over    
    
    def is_collision(self) -> bool:
        
        if (self.head.x > self.width or
            self.head.y > self.height or
            self.head.x < 0 or
            self.head.y < 0):
            return True
        if self.head in self.snake[1:]:
            return True
        return False
        
    def usr_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                if event.key == pygame.K_UP:
                    self.direction = Direction.UP
                
    # private
    def _get_direction(self, action) -> None:
        # right -> down -> left -> up
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_index: int = directions.index(self.direction)
        new_index: int = dir_index
        
        if action == RIGHT:
            new_index = (dir_index + 1) % 4
        elif action == LEFT:
            new_index = (dir_index - 1) % 4

        self.direction = directions[new_index]
    
    def _move(self, action) -> None:

        if AI:
            self._get_direction(action) 
            
        x: int = self.head.x
        y: int = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE                     
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE                     
        if self.direction == Direction.UP:
            y -= BLOCK_SIZE                     
        if self.direction == Direction.DOWN:
            y += BLOCK_SIZE                     
        self.head = Coord(x,y)
        
    def _screen_update(self) -> None:
        self.display.fill(BLACK)
        
        for block in self.snake:
            pygame.draw.rect(self.display, WHITE, 
                             pygame.Rect(block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))  
        assert self.food is not None
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        score_text = font.render(f'Score: {self.score}', False,  WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()
    
    def _spawn_food(self) -> None:
        x_blocks: int = (self.width - BLOCK_SIZE) // BLOCK_SIZE            
        y_blocks: int = (self.height - BLOCK_SIZE) // BLOCK_SIZE
        
        food_x: int  = randint(0, x_blocks) * BLOCK_SIZE            
        food_y: int  = randint(0, y_blocks) * BLOCK_SIZE
        self.food = Coord(food_x, food_y)
        
        if self.food in self.snake:
            self._spawn_food()         


if __name__ == "__main__":
    game = Game(600, 400)
    AI = False
    def play(game: Game):
        game_over = False
        while not game_over:
            game_over = game.game_step(choice([STRAIGHT, RIGHT, LEFT]))

    play(game)