
import games

while True do
g = games.tictactoe()
action = input("What do you want?")
result = g.step(action, False)
observations = environment.convert0neHot(result[0])
reward = result[1]
done = result[2]