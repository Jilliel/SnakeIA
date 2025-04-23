from bot.randy import RandomSnake
from bot.clever import CleverSnake
from learning.training import Qtrainer

DELAY = 0.1

def test_random():
    snake = RandomSnake()
    snake.debug = True
    snake.delay = DELAY
    snake.run()

def test_clever():
    snake = CleverSnake()
    snake.load("weights.pth")
    snake.debug = True
    snake.delay = DELAY
    snake.run()

test_clever()