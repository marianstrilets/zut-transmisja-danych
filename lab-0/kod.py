import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-2,2, 0.01)
y = np.cos(2*np.pi * x)

fig, ax = plt.subplots()
ax.plot(x,y)

ax.set(xlabel='X', ylabel='Y',
       title='Sinus')

ax.grid()

fig.savefig("f.png")

plt.show()
