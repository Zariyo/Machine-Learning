import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.backend.topology import Pyramid, VonNeumann, Star, Ring


# Set up optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9, 'p':4, 'k': 4}
optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                    options=options, topology=Ring())

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history


# Plot!
stats = optimizer.optimize(fx.easom, iters=100)
plot_cost_history(cost_history)
plt.show()
