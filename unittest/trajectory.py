import sys
sys.path.append('/home/ggory15/git/talos-OSF')
from OCP import trajectories as traj 

import numpy as np
import pinocchio as se3

print("")
print("Test Trajectory Euclidian")
print("")

tol = 1e-5
n = 5
q_ref = np.ones(n)
zero = np.zeros(n)

traj_euclidian = traj.TrajectoryEuclidianConstant("traj_eucl", q_ref)
assert traj_euclidian.has_trajectory_ended()
assert np.linalg.norm(traj_euclidian.computeNext().pos - q_ref, 2) < tol
assert np.linalg.norm(traj_euclidian.getSample(0.0).pos - q_ref, 2) < tol

print("")
print("No error is detected")
print("")

print("")
print("Test Trajectory SE3")
print("")

M_ref = se3.SE3.Identity()
M_ref_vec = np.hstack((M_ref.translation, M_ref.rotation.flatten()))
zero = np.zeros(6)

traj_se3 = traj.TrajectorySE3Constant("traj_se3", M_ref)
assert traj_se3.has_trajectory_ended()
assert np.linalg.norm(traj_se3.computeNext().pos - M_ref_vec, 2) < tol
assert np.linalg.norm(traj_se3.getSample(0.0).pos - M_ref_vec, 2) < tol


traj_sample = traj.TrajectorySample(12, 6)
traj_sample = traj_se3.getLastSample()
assert np.linalg.norm(traj_sample.pos - M_ref_vec, 2) < tol
assert np.linalg.norm(traj_sample.vel - zero, 2) < tol
assert np.linalg.norm(traj_sample.acc - zero, 2) < tol

print("")
print("No error is detected")
print("")
