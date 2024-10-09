import numpy as np
import matplotlib.pyplot as plt

from wzk import sql2

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/data/StaticArm04_2.db"  # download and update file path
sql2.summary(file=file)


n_voxels = 64
voxel_size = 2.10 / 64     # in m
extent = [-1.05, +1.05, -1.05, +1.05]  # in m
limb_length = 0.25  # m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 2
n_dof = 4
n_paths = sql2.get_n_rows(file=file, table="paths")
n_worlds = sql2.get_n_rows(file=file, table="worlds")

worlds = sql2.get_values(file=file, table="worlds", columns="img_cmp")
worlds = sql2.compressed2img(img_cmp=worlds, shape=(n_voxels, n_voxels), dtype=bool)

rows = np.arange(100000)
i_world, q = sql2.get_values(file=file, table="paths", rows=rows, columns=["world_i32", "q_f32"])
q = q.reshape(-1, n_waypoints, n_dof)

fig, ax = plt.subplots()
ax.plot(i_world)


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


# plot all start positions of one world
i_w = 6

fig, ax = plt.subplots()
ax.imshow(worlds[i_w].T, origin='lower', extent=extent, cmap='binary')

i_w = np.nonzero(i_world == i_w)[0]
for j in i_w:
    x = forward_kinematic(q[j])
    ax.plot(*x[0].T, color="blue", ls="-", alpha=0.5, marker="o")


# Plot and animate an example
i = 5
fig, ax = plt.subplots()

x = forward_kinematic(q[i])
ax.plot(*x[0].T, color="green", marker="o", zorder=10)
ax.plot(*x[-1].T, color="red", marker="o", zorder=10)
h = ax.plot(*x[0].T, color="blue", marker="o", zorder=20)[0]

ax.imshow(worlds[i_world[i]].T, origin="lower", extent=extent, cmap="binary")

plt.pause(1)
input()
plt.pause(1)


for t in range(n_waypoints):
    h.set_data(*x[t].T)
    plt.pause(0.1)
