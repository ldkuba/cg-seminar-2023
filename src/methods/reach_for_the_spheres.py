import gpytoolbox as gpy
import time
import numpy as np

def sdf(x):
    return sdf_at_points[np.where(sdf_points == x)[0]]

# some sdf data in numpy arrays SDF_POSITIONS, SDF_VALUES
# construct initial mesh
V0, F0 = gpy.icosphere(2)
# call our algorithm

start_time = time.time()
Vr,Fr = gpy.reach_for_the_spheres(sdf=sdf, U=sdf_points, S=sdf_at_points, V=V0, F=F0)
end_time = time.time()

results.time_ms = (end_time - start_time) * 1000.0

results.out_mesh.vertices = Vr
results.out_mesh.indices = Fr
