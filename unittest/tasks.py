import numpy as np
import pinocchio as se3

import os
import sys
sys.path.append('/home/ggory15/git/talos-OSF')
from OCP import tasks, trajectories
from scipy.spatial.transform import Rotation as R

print("")
print("Test Task COM")
print("")

tol = 1e-5
dataFolder = os.getcwd()+ '/talos_description'
urdf_filename = '/urdf/talos_full_no_grippers(package).urdf'
srdf_filename = '/srdf/talos.srdf'
base_position=[0, 0.0,1.025]
base_RPY=[0, 0, 0]

robot = se3.RobotWrapper.BuildFromURDF(dataFolder + urdf_filename, [dataFolder], se3.JointModelFreeFlyer())
model = robot.model
data = robot.data

se3.loadReferenceConfigurations(model, dataFolder + srdf_filename, False)
robot.q = model.referenceConfigurations['half_sitting'].copy()  
robot.q[:3] = base_position
robot.q[3:7] = (R.from_euler('xyz', base_RPY, degrees=False)).as_quat()
robot.v = np.zeros(robot.nv)

taskCOM = tasks.TaskComEquality("task-com", robot)
Kp = 100 * np.ones(3)
Kd = 20.0 * np.ones(3)
taskCOM.setKp(Kp)
taskCOM.setKd(Kd)

assert np.linalg.norm(Kp - taskCOM.getKp(), 2) < tol
assert np.linalg.norm(Kd - taskCOM.getKd(), 2) < tol

se3.centerOfMass(model, data, robot.q0)
com_ref = data.com[0] + np.ones(3) * 0.02
traj = trajectories.TrajectoryEuclidianConstant("traj_se3", com_ref)
sample = trajectories.TrajectorySample(0)

t = 0.0
dt = 0.001
max_it = 1001
Jpinv = np.zeros((robot.nv, 3))
error_past = 1e100
robot.v =np.zeros(robot.nv)

for i in range(0, max_it):
    se3.computeAllTerms(model, data, robot.q, robot.v)
    sample = traj.computeNext()
    taskCOM.setReference(sample)
    const = taskCOM.compute(t, robot.q, robot.v)

    Jpinv = np.linalg.pinv(const.getMatrix(), 1e-5)
    dv = np.dot(Jpinv, const.getVector().T)

    assert np.linalg.norm(np.dot(Jpinv, const.getMatrix()), 2) - 1.0 < tol
    robot.v += dt * dv
    robot.q = se3.integrate(model, robot.q, dt * robot.v)
    t += dt

    error = np.linalg.norm(taskCOM.position_error(), 2)
    assert error - error_past < 1e-4
    error_past = error
    if error < 1e-8:
        print("Success Convergence")
        break
    if i % 100 == 0:
        print("Time :", t, "COM pos error :", error, "COM vel error :", np.linalg.norm(taskCOM.velocity_error(), 2))


print("")
print("Test Task Joint Posture")
print("")

robot.q = model.referenceConfigurations['half_sitting'].copy()  
robot.q[:3] = base_position
robot.q[3:7] = (R.from_euler('xyz', base_RPY, degrees=False)).as_quat()
robot.v = np.zeros(robot.nv)
na = robot.nv - 6

taskJoint = tasks.TaskJointPosture("task-joint", robot)
Kp = 100 * np.ones(na)
Kd = 20.0 * np.ones(na)
taskJoint.setKp(Kp)
taskJoint.setKd(Kd)

assert np.linalg.norm(Kp - taskJoint.getKp(), 2) < tol
assert np.linalg.norm(Kd - taskJoint.getKd(), 2) < tol

q_ref = np.random.randn(na)
traj = trajectories.TrajectoryEuclidianConstant("traj_joint", q_ref)
sample = trajectories.TrajectorySample(0)

error_past = 1e100
t = 0.0
max_it = 10000

for i in range(0, max_it):
    se3.computeAllTerms(model, data, robot.q, robot.v)
    sample = traj.computeNext()
    taskJoint.setReference(sample)
    const = taskJoint.compute(t, robot.q, robot.v)
    
    Jpinv = np.linalg.pinv(const.getMatrix(), 1e-5)
    dv = np.dot(Jpinv, const.getVector().T)

    assert np.linalg.norm(np.dot(Jpinv, const.getMatrix()), 2) - 1.0 < tol
    robot.v += dt * dv
    robot.q = se3.integrate(model, robot.q, dt * robot.v)
    t += dt

    error = np.linalg.norm(taskJoint.position_error(), 2)
    assert error - error_past < 1e-4
    error_past = error
    if error < 1e-8:
        print("Success Convergence")
        break
    if i % 100 == 0:
        print("Time :", t, "Joint pos error :", error, "Joint vel error :",
              np.linalg.norm(taskJoint.velocity_error(), 2))

print("")
print("Test Task SE3")
print("")
robot.q = model.referenceConfigurations['half_sitting'].copy()  
robot.q[:3] = base_position
robot.q[3:7] = (R.from_euler('xyz', base_RPY, degrees=False)).as_quat()
robot.v = np.zeros(robot.nv)

taskSE3 = tasks.TaskSE3Equality("task-se3", robot, "arm_left_7_link")
# taskSE3.useLocalFrame(False)  #Use World Frame
na = 6
Kp = 100 * np.ones(na)
Kd = 20.0 * np.ones(na)
taskSE3.setKp(Kp)
taskSE3.setKd(Kd)

assert np.linalg.norm(Kp - taskSE3.getKp(), 2) < tol
assert np.linalg.norm(Kd - taskSE3.getKd(), 2) < tol

M_ref = se3.SE3.Random()
traj = trajectories.TrajectorySE3Constant("traj_se3", M_ref)
sample = trajectories.TrajectorySample(0)

t = 0.0
max_it = 2000
error_past = 1e100

for i in range(0, max_it):
    se3.computeAllTerms(model, data, robot.q, robot.v)
    sample = traj.computeNext()
    taskSE3.setReference(sample)
    const = taskSE3.compute(t, robot.q, robot.v)

    Jpinv = np.linalg.pinv(const.getMatrix(), 1e-5)
    dv = np.dot(Jpinv, const.getVector().T)

    assert np.linalg.norm(np.dot(Jpinv, const.getMatrix()), 2) - 1.0 < tol

    robot.v += dt * dv
    robot.q = se3.integrate(model, robot.q, dt * robot.v)
    t += dt

    error = np.linalg.norm(taskSE3.position_error(), 2)
    assert error - error_past < 1e-4
    error_past = error
    if error < 1e-8:
        print("Success Convergence")
        break
    if i % 100 == 0:
        print("Time :", t, "EE pos error :", error, "EE vel error :", np.linalg.norm(taskSE3.velocity_error(), 2))

