import numpy as np
import matplotlib.pyplot as plt

from wzk import sql2, trajectory


file = '/Users/jote/Documents/code/python/misc2/RobotPathData/SingleSphere02_all.db'
# file = '/Users/jote/Documents/Code/Python/RobotPathData/SingleSphere02_one-world.db'
# TODO change to you own file path


n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 2
n_paths_per_world = 1000
n_worlds = 5000


worlds = sql2.get_values_sql(file=file, table="worlds", return_type="df")
print(worlds.head())
obstacle_images = sql2.compressed2img(img_cmp=worlds.img_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)

# always 1000 paths belong to one world
# 0...999     -> world 0
# 1000...1999 -> world 1
# 2000...2999 -> world 2

batch_size = 32
n_total = n_paths_per_world * n_worlds
# batch_idx =     [0, 1, 2, 1000, 2000, 3500]
path_idx_for_batch = np.random.choice(np.arange(n_total), size=batch_size, replace=False)
path_idx_for_whole_dataset = np.arange(10000)

paths = sql2.get_values_sql(file=file, table='paths', rows=path_idx_for_whole_dataset, return_type="df")
print(paths.head())

batch_i_world = paths.world_i32.values
obstacle_images_batch = obstacle_images[batch_i_world]

q_paths = sql2.object2numeric_array(paths.q_f32.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dim)

i0 = 100
i1 = 1

i_world0 = paths.world_i32.values[i0]
i_world1 = paths.world_i32.values[i1]

fix, ax = plt.subplots()
ax.plot(paths.world_i32.values)



# ax.plot(*q_paths[i0].T, color='red', marker='o')


i_w = 5
i_s = 5
ii = np.nonzero(np.logical_and(paths.world_i32.values == i_w, paths.sample_i32.values == i_s))[0]

fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_w].T, origin='lower', extent=extent, cmap='binary')
for i in ii:
    ax.plot(*q_paths[i].T, color='blue', marker='o')


q = sql2.get_values_sql(file=file, table="paths", columns="q_f32")

q = q.reshape(-1, 20, 2)
q_start = q[:, 0, :]
q_end = q[:, -1, :]
print(q_start.min(axis=0))
print(q_start.max(axis=0))

print(q_end.min(axis=0))
print(q_end.max(axis=0))

print(q.shape)


fig, ax = plt.subplots()
ax.plot(*q_start.T, marker="o", ls="")

fig, ax = plt.subplots()
ax.plot(*q_end.T, marker="o", ls="")


# Adjust trajectory length
q_n32 = trajectory.get_path_adjusted(q[:100], n=32)
fig, ax = plt.subplots()
ax.plot(*q_n32[0].T, marker="o", color="blue", alpha=0.5, label="n_wp=32")
ax.plot(*q[0].T, marker="o", color="red", alpha=0.5, label="n_wp=20")
ax.legend()


print("end")