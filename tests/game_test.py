from unittest import TestCase
from typing import (
    List
)
from src.game import (
    BLOCK_SIZE,
    STRAIGHT,
    RIGHT,
    LEFT,
    Coord,
    Direction,
    Game
)


class GameTest(TestCase):

    def test_start(self) -> None:
        print("running test: test_start")
        game: Game = Game(100, 100)
        score: int = game.score
        game_step: int = game.step
        snake_head: Coord = game.head
        snake = game.snake

        self.assertEqual(score, 0)
        self.assertEqual(game_step, 0)
        self.assertEqual(snake_head, Coord(50, 50))

    def test_food_spawn(self) -> None:
        print("running test: test_food_spawn")
        width: int = 40
        height: int = 40
        game: Game = Game(width, height)

        assert game.food is not None
        for i in range(100):
            self.assertIn(game.food.x, range(width))
            self.assertIsInstance(game.food.x, int)
            self.assertIn(game.food.y, range(height))
            self.assertIsInstance(game.food.y, int)
            game.restart()

    def test_game_step(self) -> None:
        print("running test: test_game_step")
        # initial head DIR: RIGHT
        game: Game = Game(100, 100)
        head = game.head

        # (-> -> -> )
        for i in range(3):
            game.game_step(STRAIGHT)
        head_new_x = head.x + 3 * BLOCK_SIZE
        head_new_y = head.y
        self.assertEqual(game.head, Coord(head_new_x, head_new_y))
        self.assertEqual(game.direction, Direction.RIGHT)
        # (-> -> -> ^ )
        game.game_step(LEFT)
        head_new_x = head.x + 3 * BLOCK_SIZE
        head_new_y = head.y - BLOCK_SIZE
        self.assertEqual(game.head, Coord(head_new_x, head_new_y))
        self.assertEqual(game.direction, Direction.UP)

        # (-> -> -> ^ <-)
        game.game_step(LEFT)
        head_new_x = head.x + 2 * BLOCK_SIZE
        head_new_y = head.y - BLOCK_SIZE
        self.assertEqual(game.head, Coord(head_new_x, head_new_y))

        # (-> -> -> ^ <- <- <- V)
        game.game_step(STRAIGHT)
        game.game_step(STRAIGHT)
        game.game_step(LEFT)
        self.assertEqual(game.head, head)
        self.assertEqual(game.direction, Direction.DOWN)
        self.assertEqual(game.step, 8)
