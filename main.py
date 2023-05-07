import numpy as np
import matplotlib.pyplot as plt
from load import get_values_sql, compressed2img, object2numeric_array

file = '/Users/jote/Documents/Code/Python/RobotPathData/SingleSphere02.db'
# file = '/Users/jote/Documents/Code/Python/RobotPathData/StaticArm04.db'
# TODO change to you own file path


n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 22  # start + 20 inner points + end
n_dim = 2
n_paths_per_world = 1000
n_worlds = 5000


worlds = get_values_sql(file=file, table='worlds')
obstacle_images = compressed2img(img_cmp=worlds.obst_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

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

path_images = compressed2img(img_cmp=paths.path_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
start_images = compressed2img(img_cmp=paths.start_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
end_images = compressed2img(img_cmp=paths.end_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

batch_i_world = object2numeric_array(paths.i_world.values)
obstacle_images_batch = obstacle_images[batch_i_world]


# batch x pixels x pixels x 3
#                           . -> [start, end, world]
input_images3 = np.concatenate([start_images[..., np.newaxis],
                                end_images[..., np.newaxis],
                                obstacle_images_batch[..., np.newaxis]], axis=-1)

# batch x pixels x pixels x 2
#                           . -> [start | end, world]
input_images2 = np.concatenate([np.logical_or(start_images[..., np.newaxis],
                                              end_images[..., np.newaxis]),
                                obstacle_images_batch[..., np.newaxis]], axis=-1)
# paths are symmetrical, the image of the path from a to b looks identical to the path from b to a
# therefore we can not really distinguish between start and end, there is no temporal ordering

# batch x pixels x pixels x 1
output_images = path_images[..., np.newaxis]

# net(input_images3) -> output_images_pred
# loss(output_images_pred, output_images) -> backprop


fig, ax = plt.subplots(ncols=3)
ax[0].imshow(input_images3[0, :, :, 0].T, origin='lower', extent=extent, cmap='binary')
ax[0].set_xlabel("start")
ax[1].imshow(input_images3[0, :, :, 1].T, origin='lower', extent=extent, cmap='binary')
ax[1].set_xlabel("end")
ax[2].imshow(input_images3[0, :, :, 2].T, origin='lower', extent=extent, cmap='binary')
ax[2].set_xlabel("obstacle")


q_paths = object2numeric_array(paths.q_path.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dim)

# Plot an example
i = 5
i_world = paths.i_world.values[i]

fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_world].T, origin='lower', extent=extent, cmap='binary')
ax.imshow(start_images[i].T, origin='lower', extent=extent, cmap='Greens', alpha=0.4)
ax.imshow(end_images[i].T, origin='lower', extent=extent, cmap='Reds', alpha=0.4)
ax.imshow(path_images[i].T, origin='lower', extent=extent, cmap='Blues', alpha=0.2)

ax.plot(*q_paths[i].T, color='k', marker='o')
plt.show()
