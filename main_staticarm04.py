import numpy as np
import matplotlib.pyplot as plt

from load import get_values_sql, compressed2img, object2numeric_array

file = '/Users/jote/Documents/Code/Python/RobotPathData/StaticArm04.db'
# TODO change to you own file path

n_voxels = 64
voxel_size = 2.10 / 64     # in m
extent = [-1.05, +1.05, -1.05, +1.05]  # in m
limb_length = 0.25  # m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 2
n_dof = 4
n_worlds = 10000
# n_paths_per_world = ? varies

worlds = get_values_sql(file=file, table='worlds')
obstacle_images = compressed2img(img_cmp=worlds.obst_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

paths = get_values_sql(file=file, table='paths', rows=[0, 1, 2, 1000, 2000, 12345, 333333, 1000000])
q_paths = object2numeric_array(paths.q_path.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dof)

# Plot an example
i = 5
i_world = paths.i_world.values[i]


def forward_kinematic(q):
    q = np.cumsum(q, axis=-1)
    x = np.concatenate([q[..., np.newaxis], q[..., np.newaxis]], axis=-1)
    x[..., 0] = np.cos(x[..., 0])*limb_length
    x[..., 1] = np.sin(x[..., 1])*limb_length

    shape = np.array(q.shape + (2,))
    shape[-2] = 1
    x = np.concatenate([np.zeros(shape), x], axis=-2)
    x = np.cumsum(x, axis=-2)
    return x


# Plot an example
fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_world].T, origin='lower', extent=extent, cmap='binary')

x = forward_kinematic(q_paths[i])
ax.plot(*x[0].T, color='green', marker='o', zorder=10)
ax.plot(*x[-1].T, color='red', marker='o', zorder=10)
h = ax.plot(*x[0].T, color='blue', marker='o', zorder=0)[0]


plt.pause(1)
input()
plt.pause(1)


for i in range(n_waypoints):
    h.set_data(x[i].T)
    plt.pause(0.1)
