import numpy as np

from wzk import sql2


file = "/Users/jote/Documents/Code/Python/RobotPathData/SingleSphere02.db"
print("Before:")
sql2.summary(file=file)
print("---")

# # --- Worlds -----------------------------------------------------------------------------------------------------------
sql2.delete_columns(file=file, table="worlds",
                    columns=["rectangle_position", "rectangle_size", "edt_img_cmp", "obst_img_latent", "n_obstacles"])
sql2.add_column(file=file, table="worlds", column="world_i32", dtype=sql2.TYPE_INTEGER)
sql2.set_values_sql(file=file, table="worlds", columns=["world_i32"],
                    values=(np.arange(sql2.get_n_rows(file=file, table="worlds")).astype(np.int32),))

sql2.rename_columns(file=file, table="worlds", columns={"world_i32": "world_i32", "obst_img_cmp": "img_cmp"})
sql2.alter_table(file=file, table="worlds", columns=["world_i32", "img_cmp"],
                 dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_BLOB])
#
# --- Paths ------------------------------------------------------------------------------------------------------------
sql2.delete_columns(file=file, table="paths", columns=["q_start", "q_end"])
sql2.rename_columns(file=file, table="paths", columns={"i_world": "world_i32",
                                                       "i_sample": "sample_i32",
                                                       "q_path": "q_f64",
                                                       "start_img_cmp": "startimg_cmp",
                                                       "end_img_cmp": "endimg_cmp",
                                                       "path_img_cmp": "pathimg_cmp"})
sql2.alter_table(file=file, table="paths",
                 columns=["world_i32", "sample_i32", "q_f64", "startimg_cmp", "endimg_cmp", "pathimg_cmp"],
                 dtypes=[sql2.TYPE_INTEGER, sql2.TYPE_INTEGER, sql2.TYPE_BLOB, sql2.TYPE_BLOB, sql2.TYPE_BLOB, sql2.TYPE_BLOB])

# --- convert i64 to i32
sql2.rename_columns(file=file, table="paths", columns={"world_i32": "world_i64",
                                                       "sample_i32": "sample_i64"})
sql2.add_column(file=file, table="paths", column="q_f32", dtype=sql2.TYPE_BLOB)
sql2.set_values_sql(file=file, table="paths", columns=["world_i64", "world_i64", "q_f32"],
                    values=(sql2.get_values_sql(file=file, table="paths", columns=["world_i64"]).astype(np.int32).ravel(),
                            sql2.get_values_sql(file=file, table="paths", columns=["sample_i64"]).astype(np.int32).ravel(),
                            sql2.get_values_sql(file=file, table="paths", columns=["q_f64"]).astype(np.float32)))
sql2.rename_columns(file=file, table="paths", columns={"world_i64": "world_i32",
                                                       "sample_i64": "sample_i32"})

sql2.delete_columns(file=file, table="paths", columns=["q_f64"])


print("After:")
sql2.summary(file=file)
print("---")
