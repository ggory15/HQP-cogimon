import sys, os, subprocess, time
sys.path.append('/home/ggory15/git/talos-OSF')
import OCP as ocp

import numpy as np
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=500, nanstr='nan', precision=5, suppress=True, threshold=1000, formatter=None)

import pinocchio as se3
from scipy.spatial.transform import Rotation as R
import copy

import gepetto.corbaserver

VIEWER = True
print("")
print("Test HQP wigh Cogimon")
print("")

tol = 1e-5
URDF_FILENAME = "cogimon_arm.urdf"
SRDF_FILENAME = "cogimon_arm.srdf"
SRDF_SUBPATH = "/cogimon_srdf/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/cogimon_urdf/urdf/"

modelPath = "/home/ggory15/catkin_ws/src/cogimon"
# Load URDF file
robot = se3.RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH + URDF_FILENAME, [modelPath])
model = robot.model
data = robot.data

q0 = np.deg2rad(np.array([45, 10, 0, -110, 0, -30, 0]))
na = robot.nq
robot.q = q0
robot.v = np.zeros(robot.nv)

if VIEWER:
    l = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
    if int(l[1]) == 0:
        os.system('gepetto-gui &')
    time.sleep(1)

    cl = gepetto.corbaserver.Client()
    robot.initViewer(windowName="GSF", loadModel=True)
    robot.display(robot.q)
    time.sleep(1)

# Joint Limit Task
taskLimit = ocp.tasks.TaskJointBounds("joint-limit", robot)
ub = np.array([0, 0, 0, 0, np.deg2rad(0) + 0.01, 0, 0])
lb = np.array([0, 0, 0, 0, np.deg2rad(0) - 0.01, 0, 0])
Kp = 400 * np.ones(robot.nv)
Kd = 40.0 * np.ones(robot.nv)
taskLimit.setKp(Kp)
taskLimit.setKd(Kd)
taskLimit.setReference(lb, ub)
taskLimit.setMask(np.array([0, 0, 0, 0, 1, 0, 0]))

# SE3 Task
taskSE3 = ocp.tasks.TaskSE3Equality("task-se3", robot, "LWrMot3")
Kp = 100 * np.ones(6)
Kd = 20.0 * np.ones(6)
taskSE3.setKp(Kp)
taskSE3.setKd(Kd)
taskSE3.setMask(np.array([1, 1, 1, 1, 1, 1])) # Position Control

# SE3 Trajectory
M_ref = se3.SE3(robot.framePlacement(robot.q, robot.model.getFrameId("LWrMot3")).rotation, robot.framePlacement(q0, robot.model.getFrameId("LWrMot3")).translation + np.array([0.2, 0.4, 0.2]))
trajSE3 = ocp.trajectories.TrajectorySE3Constant("traj_se3", M_ref)
sampleSE3 = ocp.trajectories.TrajectorySample(0)

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
invdyn.addMotionTask(taskLimit, 1.0, 0, 0.0)
invdyn.addMotionTask(taskSE3, 1.0, 1, 0.0)
invdyn.addMotionTask(taskJoint, 1.0, 2, 0.0)

t = 0.0
dt = 0.001
max_it = 2000

Solver =  ocp.solvers.HQPSolver("HQP Solver")

for i in range(0, max_it):
    sampleSE3 = trajSE3.computeNext()
    taskSE3.setReference(sampleSE3)
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

    SE3error = np.linalg.norm(taskSE3.position_error(), 2)
    Slackerror = np.linalg.norm(sol[1].w, 2)
    if i%100 == 0:
        print("Time :", t, "SE3 pos error :", SE3error, "Slack SE3 : ", Slackerror)
        print("q5 :", robot.q[4])
    if VIEWER:
        robot.display(robot.q)
