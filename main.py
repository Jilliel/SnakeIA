from learning.training import Qtrainer
from bot.clever import CleverSnake
import matplotlib.pyplot as plt

# Objects definition
snake = CleverSnake()
snake.load("weights.pth")
snake.maxround = 1500
trainer = Qtrainer(snake)
trainer.epsilon_inf = 0.05
history = []

# Iterations data
epoch = 500
epochsize = 50
# Does main thing
for i in range(epoch):
    mscore = 0
    for _ in range(epochsize):
        mscore += trainer.game() / epochsize
    print(f"Epoch {i}: {mscore:0.1f}")
    history.append(mscore)

#Stores the weights
trainer.save("weights2.pth")

#Création d'un graphe
#plt.plot(history)
#plt.xlabel("N° Epoch")
#plt.ylabel("Score moyen")
#plt.savefig("history.png")
