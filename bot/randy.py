from random import randint
from bot.abstract import AbstractSnake

class RandomSnake(AbstractSnake):
    """
    Snake jouant de façon aléatoire.
    """
    def __init__(self, head):
        super().__init__(head)
    
    def play(self):
        """
        Permet au Snake de changer de direction.
        """
        move = randint(0, 2)
        match move:
            case 0:
                self.right()
            case 1:
                self.left()
            case 2:
                pass
        