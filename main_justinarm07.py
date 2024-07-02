import numpy as np

from wzk import sql2
from rokin import robots, vis

file = "/Users/jote/Documents/code/python/misc2/RobotPathData/JustinArm07.db"
# TODO change to you own file path

# TODO update db file to new format
sql2.summary(file=file)

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

worlds = sql2.get_values_sql(file=file, table="worlds", rows=[0, 1, 10], return_type="df")
obstacle_images = sql2.compressed2img(img_cmp=worlds.img_cmp.values, shape=world_shape, dtype=bool)  # True == Obstacle


# i_world = sql2.get_values_sql(file=file, table="paths", rows=-1, columns="world_i32")
# b_world1000 = i_world == 1000
# paths1000 = sql2.get_values_sql(file=file, table="paths", rows=b_world1000, return_type="df")


paths = sql2.get_values_sql(file=file, table="paths",
                            rows=[0, 1, 2,
                                  1000, 1001, 1002,
                                  10000, 10001, 10002], return_type="df")
q_paths = sql2.object2numeric_array(paths.q_f32.values)
q_paths = q_paths.reshape(-1, n_waypoints, n_dof)


# w, q = sql2.get_values_sql(file=file, table="paths", columns=["world_i32", "q_f32"],
#                             rows=[0, 1, 2,
#                                   1000, 1001, 1002,
#                                   10000, 10001, 10002],)


# TODO add blured version of full path to example

# TODO upload dataset
#  extend for single world


# add obstacle_distance for 3d robots


robot = robots.JustinArm07()
# alternative: three_pv - pyvista; mc meshcat
vis.three_mc.animate_path(robot=robot, q=q_paths[0],
                          kwargs_robot=dict(color="red", alpha=0.2),
                          kwargs_world=dict(img=obstacle_images[2], limits=limits, color="yellow"))
input()

# vis.three_pv.animate_path(robot=robot, q=q_paths[0],
#                           kwargs_world=dict(img=obstacle_images[2], limits=limits))

# move through animation with arrow keys
