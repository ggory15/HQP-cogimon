# __author__ = "Sanghyun Kim"
# __copyright__ = "Copyright (C) 2020 Sanghyun Kim"


import numpy as np
import copy
from .HQP_abstract import *
from .quadprogWrapper import *
from .HQP_output import *

class HQPSolver(HQPBase):
    def __init__(self, name):
        HQPBase.__init__(self, name)
        self.level = 0
        self.slack_boundary = 5e5
        self.ratio = 1e6 # between qddot and slack

    def resize(self, n, neq, nin, nbound):
        self.A = []
        self.b = []
        self.G = []
        self.h = []
        self.slack = np.zeros(self.level)

        for i in range(0, self.level):
            self.slack = self.slack.astype(int)
            assert neq[i] + nin[i] + nbound[i] > 0
            self.slack[i] = int(neq[i] + nin[i] + nbound[i])
            # print ("")
            if sum(neq[:i+1]) is not 0:
                self.A.append(np.zeros((int(sum(neq[:i+1])), int(n + self.slack[i]))))
                # print (self.A[i].shape[0], self.A[i].shape[1], "Asize")
                self.b.append(np.zeros(int(sum(neq[:i+1]))))
                # print (self.b[i].shape[0], "bsize")
            self.G.append(np.zeros((int(2 * sum(nin[:i+1]) + 2 * sum(nbound[:i+1])+ self.slack[i]*2), int(n + self.slack[i]))))
            # print (self.G[i].shape[0], self.G[i].shape[1], "Gsize")
            self.h.append(np.zeros(int(2 * sum(nin[:i+1]) + 2 * sum(nbound[:i+1])+ self.slack[i]*2)))
            # print (self.h[i].shape[0], "hsize")

        return True

    def solve(self, HQPData):
        problemData = HQPData.getData()
        self.level = HQPData.getSize()

        Output = []
        for i in range(self.level):
            Output.append(HQPOutput())
        cl0 = problemData[0].getData()
        n = cl0[0][1].getCols()

        neq = np.zeros(self.level)
        nin = np.zeros(self.level)
        nbound = np.zeros(self.level)

        for i in range(0, self.level):
            cl = problemData[i].getData()
            for j in range(0, len(cl)):
                assert n == cl[j][1].getCols()
                if cl[j][1].isEquality():
                    neq[i] += cl[j][1].getRows()
                elif cl[j][1].isInequality():
                    nin[i] += cl[j][1].getRows()
                else:
                    nbound[i] += cl[j][1].getRows()
        neq = neq.astype(int)
        nin = nin.astype(int)
        nbound = nbound.astype(int)
        self.resize(n, neq, nin, nbound)

        i_eq = 0
        i_in = 0
        # Set Constraint Matrix
        for i in range(0, self.level):
            cl = problemData[i].getData()
            c_eq = 0
            c_in = 0
            for j in range(0, len(cl)):
                i_help = c_eq + c_in
                if cl[j][1].isEquality():
                    self.A[i][i_eq: i_eq + cl[j][1].getRows(), 0:n] = cl[j][1].getMatrix()
                    self.A[i][i_eq: i_eq + cl[j][1].getRows(), n + i_help: n + i_help + cl[j][1].getRows()] = np.identity(cl[j][1].getRows())
                    self.b[i][i_eq: i_eq + cl[j][1].getRows()] = cl[j][1].getVector()
                    i_eq += cl[j][1].getRows()
                    c_eq += cl[j][1].getRows()
                elif cl[j][1].isInequality():
                    self.G[i][i_in: i_in + cl[j][1].getRows(), 0:n] = cl[j][1].getMatrix()
                    self.G[i][i_in: i_in + cl[j][1].getRows(), n + i_help: n + i_help  + cl[j][1].getRows()] = np.identity(cl[j][1].getRows())
                    self.G[i][i_in + cl[j][1].getRows(): i_in + 2*cl[j][1].getRows(), 0:n] = -cl[j][1].getMatrix()
                    self.G[i][i_in + cl[j][1].getRows(): i_in + 2*cl[j][1].getRows(), n + i_help: n + i_help  + cl[j][1].getRows()] = -np.identity(cl[j][1].getRows())
                    self.h[i][i_in: i_in + cl[j][1].getRows()] = cl[j][1].getUpperBound()
                    self.h[i][i_in+ cl[j][1].getRows(): i_in + 2*cl[j][1].getRows()] = -cl[j][1].getLowerBound()
                    i_in += 2 * cl[j][1].getRows()
                    c_in += cl[j][1].getRows()
                elif cl[j][1].isBound():
                    self.G[i][i_in: i_in + cl[j][1].rows(), 0:n] = np.identity(cl[j][1].rows())
                    self.G[i][i_in: i_in + cl[j][1].rows(), n + i_help: n + i_help] = np.identity(cl[j][1].rows())
                    self.G[i][i_in + cl[j][1].rows(): i_in + 2*cl[j][1].rows(), 0:n] = -np.identity(cl[j][1].rows())
                    self.G[i][i_in + cl[j][1].rows(): i_in + 2*cl[j][1].rows(), n + i_help: n + i_help] = -np.identity(cl[j][1].rows())
                    self.h[i][i_in: i_in + cl[j][1].rows()] = cl[j][1].getUpperBound()
                    self.h[i][i_in+ cl[j][1].rows(): i_in + cl[j][1].rows()] = -cl[j][1].getLowerBound()
                    i_in += 2 * cl[j][1].rows()
                    c_in += cl[j][1].rows()
            if i > 0:
                p_eq = 0
                p_in = 0
                for k in range(0, i):
                    p_cl = problemData[k].getData()
                    pc_eq = 0
                    pc_in = 0
                    for l in range(0, len(p_cl)):
                        i_help = pc_eq + pc_in
                        if p_cl[l][1].isEquality():
                            self.A[i][p_eq: p_eq+p_cl[l][1].getRows(), 0:n] = p_cl[l][1].getMatrix()
                            self.b[i][p_eq: p_eq+p_cl[l][1].getRows()] = p_cl[l][1].getVector() - Output[k].w[i_help:i_help+p_cl[l][1].getRows()]
                            p_eq += p_cl[l][1].getRows()
                            pc_eq += p_cl[l][1].getRows()
                        elif p_cl[l][1].isInequality():
                            self.G[i][p_in: p_in + p_cl[l][1].getRows(), 0:n] = p_cl[l][1].getMatrix()
                            self.G[i][p_in + p_cl[l][1].getRows(): p_in + 2*p_cl[l][1].getRows(), 0:n] = -p_cl[l][1].getMatrix()
                            self.h[i][p_in: p_in + p_cl[l][1].getRows()] = p_cl[l][1].getUpperBound() - Output[k].w[i_help:i_help+p_cl[l][1].getRows()]
                            self.h[i][p_in+ p_cl[l][1].getRows(): p_in + 2*p_cl[l][1].getRows()] = -p_cl[l][1].getLowerBound() + Output[k].w[i_help:i_help+p_cl[l][1].getRows()]
                            p_in += 2*p_cl[l][1].getRows()
                            pc_in += p_cl[l][1].getRows()
                        elif p_cl[l][1].isBound():
                            self.G[i][p_in: p_in + p_cl[l][1].rows(), 0:n] = np.identity(p_cl[l][1].rows())
                            self.G[i][p_in +p_cl[l][1].rows(): p_in + 2*p_cl[l][1].rows(), 0:n] = -np.identity(p_cl[l][1].rows())
                            self.h[i][p_in: p_in + p_cl[l][1].rows()] = p_cl[l][1].getUpperBound() -Output[k].w[i_help:i_help+p_cl[l][1].rows()]
                            self.h[i][p_in+ p_cl[l][1].rows(): p_in + 2*p_cl[l][1].rows()] = -p_cl[l][1].getLowerBound() + Output[k].w[i_help:i_help+p_cl[l][1].rows()]
                            p_in += 2 * p_cl[l][1].rows()
                            pc_in += p_cl[l][1].rows()

            # Set Slack Matrix
            self.G[i][i_in: i_in+self.slack[i], n: ] = np.identity(self.slack[i])
            self.G[i][i_in + self.slack[i]: i_in+2*self.slack[i], n: ] = -np.identity(self.slack[i])
            self.h[i][i_in:i_in+self.slack[i] ] = self.slack_boundary * np.ones(self.slack[i])
            self.h[i][i_in+self.slack[i]: ] = self.slack_boundary  * np.ones(self.slack[i])

            P = np.identity(n + self.slack[i])
            P[n:, n:] = self.ratio * np.identity(self.slack[i])

            G = self.G[i]
            h = self.h[i]
            A = self.A[i]
            b = self.b[i]

            res = quadprog_solve_qp(P, None, G, h, A, b)

            Output[i].x = res[:n]
            Output[i].w = res[n:]

        return Output
