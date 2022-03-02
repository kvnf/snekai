from unittest import TestCase
import numpy as np
from game import (
    STRAIGHT,
    LEFT,
    RIGHT,
    Coord,
    Game
)
from agent import Agent


class AgentTest(TestCase):

    def test_state_food(self) -> None:
        print("running test: test_state_food")
        game = Game(100, 100)
        agent = Agent()

        # state -> [collision forward, left, right, food up, left, down,right]

        game.food = Coord(75, 25)
        expected_state_ru: np.ndarray = np.array(
            [0, 0, 0, 1, 0, 0, 1], dtype=int)
        for i in range(len(expected_state_ru)):
            self.assertEqual(agent.state(game)[i], expected_state_ru[i])

        game.food = Coord(25, 25)
        expected_state_lu: np.ndarray = np.array(
            [0, 0, 0, 1, 0, 1, 0], dtype=int)
        for i in range(len(expected_state_lu)):
            self.assertEqual(agent.state(game)[i], expected_state_lu[i])

        game.food = Coord(25, 75)
        expected_state_ld: np.ndarray = np.array(
            [0, 0, 0, 0, 1, 1, 0], dtype=int)
        for i in range(len(expected_state_ld)):
            self.assertEqual(agent.state(game)[i], expected_state_ld[i])

        game.food = Coord(75, 75)
        expected_state_rd: np.ndarray = np.array(
            [0, 0, 0, 0, 1, 0, 1], dtype=int)
        for i in range(len(expected_state_rd)):
            self.assertEqual(agent.state(game)[i], expected_state_rd[i])

    def test_state_danger(self) -> None:
        print("running test: test_state_danger")
        game = Game(100, 100)
        agent = Agent()
        for i in range(2):
            game.game_step(STRAIGHT)

        # expect danger straight
        expected_danger_state: np.ndarray = np.array([1, 0, 0])
        for i in range(len(expected_danger_state)):
            self.assertEqual(agent.state(game)[i], expected_danger_state[i])

        game.game_step(LEFT)
        game.game_step(STRAIGHT)

        # expect danger straight - right
        expected_danger_state: np.ndarray = np.array([1, 0, 1])
        for i in range(len(expected_danger_state)):
            self.assertEqual(agent.state(game)[i], expected_danger_state[i])

        game.game_step(LEFT)
        game.game_step(STRAIGHT)
        game.game_step(STRAIGHT)
        game.game_step(STRAIGHT)
        # expect danger straight-right
        expected_danger_state: np.ndarray = np.array([1, 0, 1])
        for i in range(len(expected_danger_state)):
            self.assertEqual(agent.state(
                game)[i], expected_danger_state[i])

        game.game_step(LEFT)
        # expect danger right
        expected_danger_state: np.ndarray = np.array([0, 0, 1])
        for i in range(len(expected_danger_state)):
            self.assertEqual(agent.state(
                game)[i], expected_danger_state[i])

    def test_select_action_high(self) -> None:
        print("running test: test_select_action_high eps_threshold")
        game = Game(50, 50)
        agent = Agent(decay=200)
        state = agent.state(game)
        action = agent.select_action(state)
        self.assertIn(action, (STRAIGHT, LEFT, RIGHT))

    def test_select_action_low(self) -> None:
        print("running test: test_select_action_low eps_threshold")
        game = Game(50, 50)
        agent = Agent(200)
        agent.games_played = 1000
        game.game_step(STRAIGHT)
        state = agent.state(game)
        action = agent.select_action(state)
        self.assertIn(action, (STRAIGHT, LEFT, RIGHT))
