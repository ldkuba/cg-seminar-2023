import gpytoolbox as gpy
import time
import numpy as np
import skimage.measure as measure

def sdf(x):
    return sdf_at_points[np.where(sdf_points == x)[0]]

# construct initial mesh from coarse marching cubes
sdf_volume = np.reshape(sdf_at_points, (resolution[0], resolution[1], resolution[2]))
V0, F0, N0, _ = measure.marching_cubes(sdf_volume, level=0, spacing=(spacing, spacing, spacing), gradient_direction='ascent')

V0 = V0 + np.array([bounds[0], bounds[1], bounds[2]])

# call our algorithm
start_time = time.time()
Vr,Fr = gpy.reach_for_the_spheres(sdf=None, U=sdf_points, S=sdf_at_points, V=V0, F=F0)
end_time = time.time()

results.time_ms = (end_time - start_time) * 1000.0

results.out_mesh.vertices = Vr
results.out_mesh.indices = Fr
