import numpy as np

from wzk import sql2, mpl2
from rokin import robots, vis

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/data/JustinArm07.db"
# change to your own file path

sql2.summary(file)

n_voxels = 64
world_shape = (n_voxels, n_voxels, n_voxels)
limits = np.array([[-1.25, +1.25],
                   [-1.25, +1.25],
                   [-1.25, +1.25]])

n_waypoints = 20  # start + 20 inner points + end
n_dim = 3
n_dof = 7
n_worlds = 12500

# n_paths_per_world = ? varies

worlds_cmp = sql2.get_values(file=file, table="worlds", columns="img_cmp", rows=range(100))
worlds = sql2.compressed2img(img_cmp=worlds_cmp, shape=world_shape, dtype=bool)  # True == Obstacle


# i_world = sql2.get_values(file=file, table="paths", rows=-1, columns="world_i32")
# fig, ax = mpl2.new_fig()
# ax.plot(i_world, ls="", marker="o")


# b_world1000 = i_world == 1000
# paths1000 = sql2.get_values(file=file, table="paths", rows=b_world1000, return_type="df")

i_world, q = sql2.get_values(file=file, table="paths", columns=["world_i32", "q_f32"], rows=range(5000))
q = np.reshape(q, (-1, n_waypoints, n_dof))


# add obstacle_distance for 3d robots

j = 0
robot = robots.JustinArm07()
# alternative: vis.three_pv - pyvista; vis.three_mc - meshcat
vis.three_mc.animate_path(robot=robot, q=q[j],
                          kwargs_robot=dict(color="red", alpha=0.2),
                          kwargs_world=dict(img=worlds[i_world[j]], limits=limits, color="yellow"))
input()

# vis.three_pv.animate_path(robot=robot, q=q_paths[0],
#                           kwargs_world=dict(img=obstacle_images[2], limits=limits))

# move through animation with arrow keys
