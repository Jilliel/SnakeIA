from bot.abstract import AbstractSnake
from random import randint

class RandomSnake(AbstractSnake):
    def play(self):
        """
        Joue de façon aléatoire.
        """
        move = randint(0, 2)
        match move:        
            case 0:
                self.right()
            case 1:
                self.left()
            case _:
                pass