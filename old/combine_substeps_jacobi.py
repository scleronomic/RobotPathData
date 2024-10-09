import numpy as np
from wzk import trajectory

n_waypoints = 20
n_substeps = 5
n_dof = 2

x = np.random.random((n_waypoints, n_dof))
x_fine = trajectory.get_substeps(x=x, n=n_substeps)

do_dxfine = objective_function(x_fine)
do_dx = trajectory.combine_d_substeps__dx(d_dxs=do_dxfine, n=n_substeps)
