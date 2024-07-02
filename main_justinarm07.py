import numpy as np

from wzk import sql2
from rokin import robots, vis

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/JustinArm07.db"
# TODO change to you own file path

# TODO update db file to new format
sql2.summary(file=file)



n_voxels = 64
world_shape = (n_voxels, n_voxels, n_voxels)


voxel_size = 2.10 / 64     # in m
extent = [-1.05, +1.05, -1.05, +1.05]  # in m
limb_length = 0.25  # m
n_waypoints = 20  # start + 20 inner points + end
n_dim = 3
n_dof = 7
n_worlds = 10000


limits = np.array([[-1.25, +1.25],
                   [-1.25, +1.25],
                   [-1.25, +1.25]])

# n_paths_per_world = ? varies

worlds = sql2.get_values_sql(file=file, rows=[0, 1, 10], table="worlds", return_type="df")
obstacle_images = sql2.compressed2img(img_cmp=worlds.img_cmp.values, shape=world_shape, dtype=bool)

paths = sql2.get_values_sql(file=file, table="paths",
                            rows=[0, 1, 2,
                                  1000, 1001, 1002,
                                  10000, 10001, 10002], return_type="df")
q_paths = sql2.object2numeric_array(paths.q_f32.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dof)


robot = robots.JustinArm07()

vis.three_mc.animate_path(q=q_paths[0], robot=robot,
                          kwargs_world=dict(img=obstacle_images[2], limits=limits))

input()