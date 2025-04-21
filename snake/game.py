from bot.abstract import AbstractSnake
from random import randint
from time import sleep
import os

DEFAULT_WIDTH = 15
DEFAULT_HEIGHT = 15

position = tuple[int, int]

class Game:
    """
    Représente le jeu Snake.
    """
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT) -> None:
        # Board data
        self.width = width
        self.height = height
        self.center = (height//2 + 1, width//2 + 1)
        # Player data
        self.snake: AbstractSnake = None
        self.apple: position = None
        self.score: int = 1
        # Cycle data
        self.done: bool = False
        # Other
        self.debug: bool = False

    def addSnake(self, snake: AbstractSnake) -> None:
        """
        Change le Snake.
        """
        self.snake = snake(head=self.center)

    def getApple(self) -> position: #TODO: Very inefficient if the snake is huge.
        """
        Crée une pomme.
        """
        y = randint(0, self.height-1)
        x = randint(0, self.width-1)
        newapple = (y, x)
        if newapple in self.snake.body:
            return self.getApple()
        else:
            return newapple

    def clear(self):
        """
        Remet à zéro la partie.
        """
        self.score = 0
        self.snake.clear(head=self.center)
        self.apple = self.getApple()
        
    def run(self):
        """
        Joue une partie.
        """
        assert self.snake is not None
        # On repart de zéro.
        self.clear()
        # Puis on lance la partie.
        while not self.done:

            # Fait jouer le Snake.
            self.snake.play()
            self.snake.move()

            # On vérifie si le snake a mangé la pomme.
            head = self.snake.getHead()
            if head == self.apple:
                self.score += 1
                self.apple = self.getApple()
            else:
                self.snake.retract()

            # On vérifie le snake n'est pas rentré dans lui-même.
            if head in self.snake.getBody():
                self.done = True

            # On vérifie si le snake n'a pas collisionné un mur.
            yh, xh = head
            if yh < 0 or yh >= self.height:
                self.done = True
            if xh < 0 or xh >= self.width:
                self.done = True
            
            if self.debug:
                self.show()
                sleep(0.3)
        
    def show(self):
        """
        Affiche l'état de la partie.
        """
        os.system("clear")
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.apple:
                    symbole = "0"
                elif (i, j) == self.snake.getHead():
                    symbole = "X"
                elif (i, j) in self.snake.getBody():
                    symbole = "*"
                else:
                    symbole = "-"
                print(symbole, end=" ")
            print()
