import sys, os
sys.path.append('/home/ggory15/git/talos-OSF')
import OCP as ocp

import numpy as np
import pinocchio as se3
from scipy.spatial.transform import Rotation as R
import copy

print("")
print("Test HQP formulation")
print("")

tol = 1e-5
dataFolder = os.getcwd()+ '/talos_description'
urdf_filename = '/urdf/talos_full_no_grippers(package).urdf'
srdf_filename = '/srdf/talos.srdf'
base_position=[0, 0.0,1.025]
base_RPY=[0, 0, 0]

robot = se3.RobotWrapper.BuildFromURDF(dataFolder + urdf_filename, [dataFolder]) #Fixed model
model = robot.model
data = robot.data

se3.loadReferenceConfigurations(model, dataFolder + srdf_filename, False)
robot.q = model.referenceConfigurations['half_sitting'].copy()  
robot.v = np.zeros(robot.nv)
na = robot.model.nq

# COM Task
taskCOM = ocp.tasks.TaskComEquality("task-com", robot)
Kp = 100 * np.ones(3)
Kd = 20.0 * np.ones(3)
taskCOM.setKp(Kp)
taskCOM.setKd(Kd)

# COM Trajectory
se3.centerOfMass(model, data, robot.q0)
com_ref = data.com[0] + np.ones(3) * 0.02
trajCOM = ocp.trajectories.TrajectoryEuclidianConstant("traj_se3", com_ref)
sampleCOM = ocp.trajectories.TrajectorySample(0)

# Joint Task
taskJoint = ocp.tasks.TaskJointPosture("task-joint", robot)
Kp = 100 * np.ones(na)
Kd = 20.0 * np.ones(na)
taskJoint.setKp(Kp)
taskJoint.setKd(Kd)

q_ref = copy.deepcopy(robot.q)
trajJoint = ocp.trajectories.TrajectoryEuclidianConstant("traj_joint", q_ref)
sampleJoint = ocp.trajectories.TrajectorySample(0)

invdyn = ocp.FormulationHQP("HQP", robot, False)
invdyn.addMotionTask(taskCOM, 1.0, 0, 0.0)
invdyn.addMotionTask(taskJoint, 1.0, 1, 0.0)

t = 0.0
dt = 0.001
max_it = 2000

Solver =  ocp.solvers.HQPSolver("HQP Solver")

for i in range(0, max_it):
    sampleCOM = trajCOM.computeNext()
    taskCOM.setReference(sampleCOM)
    sampleJoint = trajJoint.computeNext()
    taskJoint.setReference(sampleJoint)
    HQPData = invdyn.computeProblemData(t, robot.q, robot.v)

    if i == 0:
        HQPData.print(False)
    
    sol = Solver.solve(HQPData)
    dv = invdyn.getAccelerations(sol)
    robot.v += dt * dv
    robot.q = se3.integrate(model, robot.q, dt * robot.v)
    t += dt

    COMerror = np.linalg.norm(taskCOM.position_error(), 2)
    Slackerror = np.linalg.norm(sol[0].w, 2)
    if i%100 == 0:
        print("Time :", t, "COM pos error :", COMerror, "Slack COM : ", Slackerror)

    assert np.linalg.norm(dv,2) < 1e6
    assert np.linalg.norm(robot.v,2) < 1e6