from unittest import TestCase
import numpy as np
from agent import Agent
from game import (
    Coord,
    Game
)
class AgentTest(TestCase):
    
    def test_state(self) -> None:
        game = Game(100, 100)
        agent = Agent()
        
        
        # state -> collision forward, left, right, food up, left, down,right 

        game.food = Coord(75, 25)
        expected_state_ru: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 1], dtype=int)
        for i in range(len(expected_state_ru)):
            self.assertEqual(agent.state(game)[i], expected_state_ru[i])
        
        
        game.food = Coord(25,25)
        expected_state_lu: np.ndarray = np.array([0, 0, 0, 1, 0, 1, 0], dtype=int)
        for i in range(len(expected_state_lu)):
            self.assertEqual(agent.state(game)[i], expected_state_lu[i])
        
        game.food = Coord(25,75)
        expected_state_ld: np.ndarray = np.array([0, 0, 0, 0, 1, 1, 0], dtype=int)
        for i in range(len(expected_state_ld)):
            self.assertEqual(agent.state(game)[i], expected_state_ld[i])
        
        game.food = Coord(25,75)
        expected_state_rd: np.ndarray = np.array([0, 0, 0, 0, 1, 0, 1], dtype=int)
        for i in range(len(expected_state_ld)):
            self.assertEqual(agent.state(game)[i], expected_state_ld[i])
    