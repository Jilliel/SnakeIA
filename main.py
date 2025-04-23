from learning.training import Qtrainer
from bot.clever import CleverSnake
import matplotlib.pyplot as plt

# Objects definition
snake = CleverSnake()
trainer = Qtrainer(snake)
history = []

# Iterations data
epoch = 100
epochsize = 50
# Does main thing
for i in range(epoch):
    mscore = 0
    for _ in range(epochsize):
        mscore += trainer.game() / epochsize
    print(f"Epoch {i}: {mscore}")
    history.append(mscore)

#Création d'un graphe
plt.plot(history)
plt.xlabel("N° Epoch")
plt.ylabel("Score moyen")
plt.savefig("history.png")

#Shows the result
input("Ready ?")
snake.debug = True
snake.run()
