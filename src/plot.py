import matplotlib.pyplot as plt
import numpy as np

data_file = "reward.txt"

f = open(data_file)
# d = [float(y) for x in f.readlines() for y in x.split(',')]
#  for getting last values from reward.txt
d = []
for x in f.readlines():
  d.append(float(x.split(',')[-1]))

x = range(len(d))
y = d
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
plt.plot(x,y, 'yo', x, fit_fn(x), 'k--')

plt.xlabel("iteration number")

if data_file == "reward.txt":
  plt.ylabel("reward")
  plt.title("reward over time")

if data_file == "distance.txt":
  plt.ylabel("distance")
  plt.title("distance away from enemy over time")

if data_file == "time.txt":
  plt.ylabel("seconds")
  plt.title("amount of time villager is alive over time")

plt.show()
