from learning.training import Qtrainer
from bot.clever import CleverSnake
from tqdm import tqdm

snake = CleverSnake()
trainer = Qtrainer(snake)

iterations = 1500
for _ in tqdm(range(iterations)):
    trainer.game()

input("Ready ?")
snake.debug = True
snake.run()