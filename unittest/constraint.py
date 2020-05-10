import sys
sys.path.append('/home/ggory15/git/talos-OSF')
from OCP import constraints as const 

import numpy as np
import pinocchio as se3

print("")
print("Test Constraint Bound")
print("")

tol = 1e-5
n = 5
lb = -1.0 * np.ones(n)
ub = np.ones(n)
ConstBound = const.ConstraintBound("bounds", lb=lb, ub=ub)

assert ConstBound.isBound()
assert not ConstBound.isEquality()
assert not ConstBound.isInequality()
assert ConstBound.rows() == n
assert ConstBound.cols() == n

assert np.linalg.norm(lb-ConstBound.getLowerBound(), 2) < tol
assert np.linalg.norm(ub-ConstBound.getUpperBound(), 2) < tol

lb *= 2.0
assert np.linalg.norm(lb - ConstBound.getLowerBound(), 2) > tol
ConstBound.setLowerBound(lb)
assert np.linalg.norm(lb-ConstBound.getLowerBound(), 2) < tol

ub *= 2.0
assert np.linalg.norm(ub - ConstBound.getUpperBound(), 2) > tol
ConstBound.setUpperBound(ub)
assert np.linalg.norm(ub-ConstBound.getUpperBound(), 2) < tol

print("")
print("Test Constraint Equality")
print("")
n = 5
m = 2
A = np.matrix(np.ones((m, n)))
b = np.matrix(np.ones(m)).transpose()
equality = const.ConstraintEquality("equality", A=A, b=b)

assert not equality.isBound()
assert equality.isEquality()
assert not equality.isInequality()

assert equality.getRows() == m
assert equality.getCols() == n

assert np.linalg.norm(A - equality.getMatrix(), 2) < tol
assert np.linalg.norm(b - equality.getVector(), 2) < tol

b *= 2.0
assert np.linalg.norm(b - equality.getVector(), 2) is not 0
equality.setVector(b)
assert np.linalg.norm(b - equality.getVector(), 2) < tol

A *= 2.0
assert np.linalg.norm(A - equality.getMatrix(), 2) is not 0
equality.setMatrix(A)
assert np.linalg.norm(A - equality.getMatrix(), 2) < tol

print("")
print("Test Constraint Inequality")
print("")

n = 5
m = 2
A = np.matrix(np.ones((m, n)))
lb = np.matrix(-np.ones(m)).transpose()
ub = np.matrix(np.ones(m)).transpose()
inequality = const.ConstraintInequality("inequality", A=A, lb=lb, ub=ub)

assert not inequality.isBound()
assert not inequality.isEquality()
assert inequality.isInequality()

assert inequality.getRows() == m
assert inequality.getCols() == n

lb *= 2.0
assert np.linalg.norm(lb - inequality.getLowerBound(), 2) is not 0
inequality.setLowerBound(lb)
assert np.linalg.norm(lb - inequality.getLowerBound(), 2) < tol

A *= 2.0
assert np.linalg.norm(A - inequality.getMatrix(), 2) is not 0
inequality.setMatrix(A)
assert np.linalg.norm(A - inequality.getMatrix(), 2) < tol

print("All test is done")
