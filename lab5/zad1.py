import pyswarms as ps
import math
import numpy as np
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

def endurance(args):
 return - (math.exp(-2*(args[1]-math.sin(args[0]))**2)+math.sin(args[2]*args[3])+math.cos(args[4]*args[5]))

def f(x):
 n_particles = x.shape[0]
 j = [endurance(x[i]) for i in range(n_particles)]
 return np.array(j)


x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=4000, dimensions=6,
options=options, bounds=my_bounds)
optimizer.optimize(f, iters=37000)

cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()


