from bot.randy import RandomSnake
from bot.clever import CleverSnake

def test_random():
    snake = RandomSnake()
    snake.debug = True
    snake.run()

def test_clever():
    snake = CleverSnake()
    snake.debug = True
    snake.run()
