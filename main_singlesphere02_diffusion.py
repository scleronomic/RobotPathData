import numpy as np
import matplotlib.pyplot as plt

from wzk import sql2, trajectory


file = '/Users/jote/Documents/code/python/misc2/RobotPathData/data/SingleSphere02_all.db'
# file = '/Users/jote/Documents/code/python/misc2/RobotPathData/data/SingleSphere02_one-world.db'
# download and update file

sql2.summary(file)

n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 20  # start + 18 inner points + end
n_dim = 2
n_paths = sql2.get_n_rows(file=file, table="paths")
n_worlds = sql2.get_n_rows(file=file, table="worlds")

obstacle_images = sql2.get_values(file=file, table="worlds", columns="img_cmp")
obstacle_images = sql2.compressed2img(img_cmp=obstacle_images, shape=(n_voxels, n_voxels), dtype=bool)

batch_size = 32
rows = np.arange(10000)

i_sample, i_world, q = sql2.get_values(file=file, table="paths", rows=rows, columns=["sample_i32", "world_i32", "q_f32"])
q = q.reshape(-1, n_waypoints, n_dim)


# plot structure of world indices
fix, ax = plt.subplots()
ax.plot(i_world)


# plot all tries to a single motion task
i_w = 5
i_s = 5
i_ws = np.nonzero(np.logical_and(i_world == i_w, i_sample == i_s))[0]

fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_w].T, origin='lower', extent=extent, cmap='binary')
for i in i_ws:
    ax.plot(*q[i].T, color='blue', marker='o')

# plot all start and end points in a single world
q_start = q[:, 0, :]
q_end = q[:, -1, :]

i_w = 6
fig, ax = plt.subplots()
ax.imshow(obstacle_images[i_w].T, origin='lower', extent=extent, cmap='binary')

i_w = np.nonzero(i_world == i_w)[0]

for i in i_w:
    ax.plot(*q_start[i].T, color='blue', marker='o', ls="")
    ax.plot(*q_end[i].T, color='red', marker='o', ls="")


# Adjust trajectory length
q_n32 = trajectory.get_path_adjusted(q[:100], n=32)
fig, ax = plt.subplots()
ax.plot(*q_n32[0].T, marker="o", color="blue", alpha=0.5, label="n_wp=32")
ax.plot(*q[0].T, marker="o", color="red", alpha=0.5, label="n_wp=20")
ax.legend()


print("end")
