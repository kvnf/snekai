from src.agent import Agent, train_agent
from src.game import Game

MEM_SIZE: int = 100000
BATCH_SIZE: int = 1000
EPS_START: float = 1.0
EPS_END: float = -0.3
LEARNING_RATE: float = 0.0001
GAMMA: float = 0.9

agent = Agent(MEM_SIZE, BATCH_SIZE, EPS_START, EPS_END, LEARNING_RATE, GAMMA)
game = Game(640, 480)
agent.state(game)
train_agent(agent, game)
