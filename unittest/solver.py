import sys
sys.path.append('/home/ggory15/git/talos-OSF')
from OCP import solvers, constraints 

import numpy as np
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=500, nanstr='nan', precision=5, suppress=False, threshold=1000, formatter=None)
import pinocchio as se3

print("")
print("Test Solvers")
print("")

EPS = 1e-3
nTest = 100
n = 60
neq = 36
nin = 40
damping = 1e-10

NORMAL_DISTR_VAR = 10.0
GRADIENT_PERTURBATION_VARIANCE = 1e-3
HESSIAN_PERTURBATION_VARIANCE = 1e-2
MARGIN_PERC = 1e-3

print("Gonna perform", nTest, "tests with", n, "variables, ", neq, "equalities", nin, "inequalities")
print("")

HQPData = solvers.HQPData()
A1 = np.random.rand(n, n) + 0.001 * np.identity(n)
b1 = np.random.rand(n)
cost = constraints.ConstraintEquality("c1", A=A1, b=b1)

x = np.dot(np.linalg.inv(A1) , b1)
A_in = np.random.rand(nin, n)
A_lb = np.random.rand(nin) * NORMAL_DISTR_VAR
A_ub = np.random.rand(nin) * NORMAL_DISTR_VAR
constrVal = np.dot(A_in , x)

for i in range(0, nin):
    if A_ub[i] < A_lb[i]:
        A_ub[i] = A_lb[i] + MARGIN_PERC * np.abs(A_lb[i])
        A_lb[i] = A_lb[i] - MARGIN_PERC * np.abs(A_lb[i])

    if constrVal[i] > A_ub[i]:
        A_ub[i] = constrVal[i] + MARGIN_PERC * np.abs(constrVal[i])
    elif constrVal[i] < A_lb[i]:
        A_lb[i] = constrVal[i] - MARGIN_PERC * np.abs(constrVal[i])

in_const = constraints.ConstraintInequality("in1", A=A_in, lb=A_lb, ub=A_ub)
A_eq = np.random.rand(neq, n)
b_eq = np.dot(A_eq, x)
eq_const = constraints.ConstraintEquality("eq1", A=A_eq, b=b_eq)

ConstraintLevel0 = solvers.ConstraintLevel()
ConstraintLevel0.setConstraint(1.0, eq_const)
ConstraintLevel0.setConstraint(1.0, in_const)
print("check constraint level #0")
ConstraintLevel0.print()
print(" ")

ConstraintLevel1 = solvers.ConstraintLevel()
ConstraintLevel1.setConstraint(1.0, cost)
print("check constraint level #1")
ConstraintLevel1.print()
print(" ")

HQPData.setData(0, ConstraintLevel0)
HQPData.setData(1, ConstraintLevel1) 
print("check HQPData")
HQPData.print(False)
print (" ")

gradientPerturbations = []
hessianPerturbations = []


for i in range(0, nTest):
    gradientPerturbations.append(np.ones(n)* GRADIENT_PERTURBATION_VARIANCE )
    hessianPerturbations.append(np.ones((n, n)) * HESSIAN_PERTURBATION_VARIANCE)

HQPSolver = solvers.HQPSolver("solver")

for i in range(0, nTest):
    cost.setMatrix(cost.getMatrix() + hessianPerturbations[i])
    cost.setVector(cost.getVector() + gradientPerturbations[i])
    HQPoutput = HQPSolver.solve(HQPData)
    
    assert (np.linalg.norm( np.dot(A_eq, HQPoutput[1].x) - b_eq, 2) < EPS)
    assert (np.dot(A_in, HQPoutput[1].x) <= A_ub + EPS).all()
    assert (np.dot(A_in, HQPoutput[1].x) >= A_lb - EPS).all()
    print (i, "Trials")




