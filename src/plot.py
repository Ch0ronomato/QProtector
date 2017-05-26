import matplotlib.pyplot as plt 
import numpy as np
f = open("reward.txt")
d = [float(y) for x in f.readlines() for y in x.split(',')]
x = range(len(d))
y = d
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.show()
