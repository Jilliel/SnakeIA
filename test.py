from bot.randy import RandomSnake
from bot.final import FinalSnake

def testRandSnake():
    snake = RandomSnake()
    snake.debug = True
    snake.run()

def testFinalSnake():
    snake = FinalSnake()
    snake.load("weights.pth")
    snake.debug = True
    snake.run()

if __name__ == "__main__":
    #testRandSnake()
    testFinalSnake()