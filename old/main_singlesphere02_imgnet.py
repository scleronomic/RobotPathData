import numpy as np

from wzk import sql2, trajectory
import matplotlib.pyplot as plt

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/data/SingleSphere02_v7.db"  # download from cloud, update path
sql2.summary(file=file)

n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 22  # start + 20 inner points + end
n_dim = 2
n_paths = sql2.get_n_rows(file=file, table="paths")
n_worlds = sql2.get_n_rows(file=file, table="worlds")


worlds = sql2.get_values(file=file, table="worlds", columns="img_cmp")
worlds = sql2.compressed2img(img_cmp=worlds, shape=(n_voxels, n_voxels), dtype=bool)

# i_world = sql2.get_values(file=file, table="paths", rows=-1, columns="world_i32")
# fig, ax = plt.subplots()
# ax.plot(i_world[:10000], ls="", marker="o")


# load 128 random paths
batch = np.random.choice(np.arange(n_paths), size=128, replace=False)  # load random paths
# - or pick the first 128
# batch = np.arange(128)
paths = sql2.get_values(file=file, table="paths", rows=batch, return_type="df")

path_images = sql2.compressed2img(img_cmp=paths.pathimg_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)
start_images = sql2.compressed2img(img_cmp=paths.startimg_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)
end_images = sql2.compressed2img(img_cmp=paths.endimg_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)
i_world = paths.world_i32.values
obstacle_images = worlds[i_world]

# batch x pixels x pixels x 3
#                           . -> [start, end, world]
input_images3 = np.concatenate([start_images[..., np.newaxis],
                                end_images[..., np.newaxis],
                                obstacle_images[..., np.newaxis]], axis=-1)

# batch x pixels x pixels x 2
#                           . -> [start | end, world]
input_images2 = np.concatenate([np.logical_or(start_images[..., np.newaxis],
                                              end_images[..., np.newaxis]),
                                obstacle_images[..., np.newaxis]], axis=-1)
# paths are symmetrical, the image of the path from a to b looks identical to the path from b to a
# therefore we can not really distinguish between start and end, there is no temporal ordering

# batch x pixels x pixels x 1
output_images = path_images[..., np.newaxis]

# net(input_images3) -> output_images_pred
# loss(output_images_pred, output_images) -> backprop


# plot example
i = 0
print(f"sample {batch[i]} | world {i_world[i]}")
# fig, ax = plt.subplots(ncols=4)
# ax[0].imshow(input_images3[i, :, :, 0].T, origin="lower", extent=extent, cmap="Greens")
# ax[0].set_xlabel("start")
# ax[1].imshow(input_images3[i, :, :, 1].T, origin="lower", extent=extent, cmap="Reds")
# ax[1].set_xlabel("end")
# ax[2].imshow(input_images3[i, :, :, 2].T, origin="lower", extent=extent, cmap="Greys")
# ax[2].set_xlabel("obstacle")
# ax[3].imshow(output_images[i, :, :, 0].T, origin="lower", extent=extent, cmap="Blues")
# ax[3].set_xlabel("obstacle")


fig, ax = plt.subplots()
ax.imshow(input_images3[i, :, :, 2].T, origin="lower", extent=extent, cmap="Greys", alpha=1.0)  # world
ax.imshow(input_images3[i, :, :, 0].T, origin="lower", extent=extent, cmap="Greens", alpha=0.4)  # start
ax.imshow(input_images3[i, :, :, 1].T, origin="lower", extent=extent, cmap="Reds", alpha=0.4)   # end
ax.imshow(output_images[i, :, :, 0].T, origin="lower", extent=extent, cmap="Blues", alpha=0.2)  # path
