from snake.game import Game
from bot.randy import RandomSnake

game = Game()
game.debug = True
game.addSnake(RandomSnake)
game.run()
