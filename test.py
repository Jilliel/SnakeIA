from bot.randy import RandomSnake
from bot.clever import CleverSnake
from learning.training import Qtrainer
def test_random():
    snake = RandomSnake()
    snake.debug = True
    snake.run()

def test_clever():
    snake = CleverSnake()
    snake.debug = True
    snake.run()

def test_train():
    snake = CleverSnake()
    trainer = Qtrainer(snake)
    trainer.play()
