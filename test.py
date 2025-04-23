from bot.randy import RandomSnake
from bot.final import FinalSnake

def testRandSnake():
    snake = RandomSnake()
    snake.debug = True
    snake.run()

def testFinalSnake():
    snake = FinalSnake()
    snake.load("weights.pth")
    snake.maxround = 2000
    snake.debug_delay = 0.05
    snake.debug = True
    snake.run()

if __name__ == "__main__":
    testFinalSnake()