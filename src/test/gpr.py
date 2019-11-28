import GPy
import GPyOpt
from numpy.random import seed

# create the object function
f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd = 0.1)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_3', 'type': 'continuous', 'domain': f_true.bods[0]},]
f_true.plot()

f_test = lambda x: x.sum()**2

# Creates three identical objects that we will later use to compare the optimization strategies
myBopt2D = GPyOpt.methods.BayesianOptimization(f_test,un
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='EI',
                                              normalize_Y = True,
                                              acquisition_weight = 2)

# runs the optimization for the three methods
max_iter = 40  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter,max_time,verbosity=False)

myBopt2D.plot_acquisition()

myBopt2D.plot_convergence()