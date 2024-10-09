import numpy as np

from wzk import sql2, trajectory
import matplotlib.pyplot as plt

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/data/SingleSphere02.db"  # download from cloud, update path
sql2.summary(file=file)

n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 2
n_dof = 2
n_paths = sql2.get_n_rows(file=file, table="paths")
n_worlds = sql2.get_n_rows(file=file, table="worlds")


worlds = sql2.get_values(file=file, table="worlds", columns="img_cmp")
worlds = sql2.compressed2img(img_cmp=worlds, shape=(n_voxels, n_voxels), dtype=bool)

i_world, q = sql2.get_values(file=file, table="paths", rows=-1, columns=["world_i32", "q_f32"])
q = q.reshape(-1, n_waypoints, n_dof)
fig, ax = plt.subplots()
ax.plot(i_world[:10000], ls="", marker="o")


# Plot all paths in a world
i_w = 6

fig, ax = plt.subplots()
ax.imshow(worlds[i_w].T, origin='lower', extent=extent, cmap='binary')

i_w = np.nonzero(i_world == i_w)[0]
for j in i_w:
    ax.plot(*q[j].T, color="blue", ls="-", alpha=0.5)





# Show how to adjust trajecoty length
i = 2017

q_n32 = trajectory.get_path_adjusted(q[i], n=32)
fig, ax = plt.subplots()
ax.imshow(worlds[i_world[i], :, :].T, origin="lower", extent=extent, cmap="Greys", alpha=1.0)  # world
ax.plot(*q_n32.T, marker="o", color="blue", alpha=0.3, label="n_wp=32")
ax.plot(*q[i].T, marker="o", color="red", alpha=0.3, label="n_wp=20")
ax.legend()
