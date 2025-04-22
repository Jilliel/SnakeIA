from bot.randy import RandomSnake
from bot.student import CleverSnake

def testRandSnake():
    snake = RandomSnake(width=15, height=15)
    snake.debug = True
    snake.run()

def testCleverSnake():
    snake = CleverSnake()
    snake.debug = True
    snake.run()

if __name__ == "__main__":
    #testRandSnake()
    testCleverSnake()