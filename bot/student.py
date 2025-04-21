from abstract import AbstractSnake

class CleverSnake(AbstractSnake):
    def __init__(self, head):
        super().__init__(head)

    def play(self):
        """
        Permet au Snake de changer de direction.
        """
        