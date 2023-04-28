import numpy as np
import matplotlib.pyplot as plt
from load import get_values_sql, compressed2img, object2numeric_array

file = '/Users/jote/Documents/Code/Python/RobotPathData/SingleSphere02_all.db'
# TODO change to you own file path


n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 2
n_paths_per_world = 1000
n_worlds = 5000


worlds = get_values_sql(file=file, table='worlds')
print(worlds.head())
obstacle_images = compressed2img(img_cmp=worlds.img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

# always 1000 paths belong to one world
# 0...999     -> world 0
# 1000...1999 -> world 1
# 2000...2999 -> world 2

batch_size = 32
n_total = n_paths_per_world * n_worlds
# batch_idx =     [0, 1, 2, 1000, 2000, 3500]
path_idx_for_batch = np.random.choice(np.arange(n_total), size=batch_size, replace=False)
path_idx_for_whole_dataset = np.arange(10000)

paths = get_values_sql(file=file, table='paths', rows=path_idx_for_whole_dataset)
print(paths.head())

batch_i_world = object2numeric_array(paths.world_i32.values)
obstacle_images_batch = obstacle_images[batch_i_world]

q_paths = object2numeric_array(paths.q_f32.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dim)

# Plot an example
i0 = 5442
i1 = 5443

i_world0 = paths.world_i32.values[i0]
i_world1 = paths.world_i32.values[i1]

fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_world0].T, origin='lower', extent=extent, cmap='binary')

ax.plot(*q_paths[i0].T, color='red', marker='o')
ax.plot(*q_paths[i1].T, color='blue', marker='o')
plt.show()
