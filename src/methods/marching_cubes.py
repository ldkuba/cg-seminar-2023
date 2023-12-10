import skimage.measure as measure
import numpy as np
import time

results = TestResults()

# reshape volume
sdf_volume = np.reshape(sdf_at_points, (resolution[0], resolution[1], resolution[2]))

# run marching cubes
start_time = time.time()
vertices, indices, normals, values = measure.marching_cubes(sdf_volume, spacing=(spacing, spacing, spacing), gradient_direction='ascent')
end_time = time.time()

results.time_ms = (end_time - start_time) * 1000.0

results.out_mesh = TriangleMesh()
results.out_mesh.vertices = vertices
results.out_mesh.indices = indices

## === visualize mesh ===
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# plt.switch_backend('TkAgg')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_trisurf(vertices[:,0], vertices[:,2], vertices[:,1], triangles=triangles, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=True)

# x = vertices[:,0]
# y = vertices[:,2]
# z = vertices[:,1]
# max_range = np.array([x.max()-x.min(), y.max()-y.min(),
#                       z.max()-z.min()]).max() / 2.0
# mean_x = x.mean()
# mean_y = y.mean()
# mean_z = z.mean()
# ax.set_xlim(mean_x - max_range, mean_x + max_range)
# ax.set_ylim(mean_y - max_range, mean_y + max_range)
# ax.set_zlim(mean_z - max_range, mean_z + max_range)
# plt.show()
