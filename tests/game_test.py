from unittest import TestCase
from typing import (
    List
)
from game import (
    BLOCK_SIZE,
    Coord,
    Game
)

STRAIGHT =   [1, 0, 0]
RIGHT   =   [0, 1, 0]
LEFT    =   [0, 0, 1]


class GameTest(TestCase):
    
    def test_start(self) -> None:
        game: Game = Game(100, 100)
        score: int = game.score
        frame: int = game.frame
        snake_head: Coord = game.head
        snake = game.snake
        
        self.assertEqual(score, 0)
        self.assertEqual(frame, 0)
        self.assertEqual(snake_head, Coord(50,50))
        self.assertEqual(snake, [snake_head])
    
    def test_food_spawn(self) -> None:
        width: int = 40
        height: int = 40
        game: Game = Game(width, height)

        for i in range(100):
            self.assertIn(game.food.x, range(width))
            self.assertIsInstance(game.food.x, int)
            self.assertIn(game.food.y, range(height))
            self.assertIsInstance(game.food.y, int)
            game.restart()
    
    def test_game_step(self) -> None:
        # initial head DIR: RIGHT
        game: Game = Game(100, 100)
        head = game.head
        
        for i in range(3):
            game.game_step(STRAIGHT)
        
        head_new_x = head.x + 3 * BLOCK_SIZE 
        head_new_y = head.y
        self.assertEqual(game.head, Coord(head_new_x, head_new_y))
        
        
         