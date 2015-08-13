'''
Sequential Minimal Optimization(SMO)
Created on Jun 5, 2015 
@author: apex

'''

import numpy as np

class SMOStruct:
    # init the structure with parameters
    def __init__(self, data_X, data_y, C, toler, kernel_tup):
        self.X = np.mat(data_X)
        self.y = np.mat(data_y).T
        self.C = C
        self.tol = toler
        self.kernel_tup = kernel_tup
        self.m = np.shape(data_X)[0]
        self.n = np.shape(data_X)[1]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.w = np.mat(np.zeros(self.n)).T
        self.e_cache = np.mat(np.zeros(self.m)).T
        # init kernel cache matrix: lin or rbf
        if kernel_tup[0] == 'lin':
            self.K = self.X * self.X.T
        elif kernel_tup[0] == 'rbf':
            self.K = np.mat(np.zeros((self.m, self.m)))
            for i in range(self.m):
                for j in range(self.m):
                    self.K[i, j] = (self.X[i] - self.X[j]) * (self.X[i] - self.X[j]).T
                    self.K[i, j] = exp(self.K[i, j]/(-1 * self.kernel_tup[1]**2))
        else:
            pass


def take_step(i1, i2, smo):
    alpha1 = smo.alphas[i1]
    y1 = smo.y[i1]
    if alpha1 > 0 and alpha1 < smo.C:
        E1 = smo.e_cache[i1]
    else:
        E1 = smo.X[i1] * smo.w + smo.b - smo.y[i1]
    alpha2 = smo.alphas[i2]
    y2 = smo.y[i2]
    E2 = smo.e_cache[i2]
    s = y1 * y2
    if y1 == y2:
        L = max(0, alpha1+alpha2-smo.C)
        H = min(smo.C, alpha1+alpha2)
    else:
        L = max(0, alpha2-alpha1)
        H = min(smo.C, smo.C+alpha2-alpha1)
    if L == H:
        return 0
    eta = smo.K[i1, i1] + smo.K[i2, i2] - 2*smo.K[i1, i2]
    if eta > 0:
        a2 = alpha2 + y2*(E1-E2)/eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        c1 = eta / 2.0
        c2 = y2 * (E1 - E2) - eta * alpha2
        Lobj = c1 * L * L + c2 * L
        Hobj = c1 * H * H + c2 * H
        if Lobj > Hobj + smo.tol:
            a2 = L
        elif Lobj < Hobj - smo.tol:
            a2 = H
        else:
            a2 = alpha2
    if abs(a2 - alpha2) < smo.tol:
        return 0
    a1 = alpha1 - s*(a2 - alpha2)
    if a1 > 0 and a1 < smo.C:
        bnew = smo.b - E1 - y1 * (a1 - alpha1) * smo.K[i1, i1] - y2 * (a2 - alpha2) * smo.K[i1, i2]
    elif a2 > 0 and a2 < smo.C:
        bnew = smo.b - E2 - y1 * (a1 - alpha1) * smo.K[i1, i2] - y2 * (a2 - alpha2) * smo.K[i2, i2]
    else:
        b1 = smo.b - E1 - y1 * (a1 - alpha1) * smo.K[i1, i1] - y2 * (a2 - alpha2) * smo.K[i1, i2]
        b2 = smo.b - E2 - y1 * (a1 - alpha1) * smo.K[i1, i2] - y2 * (a2 - alpha2) * smo.K[i2, i2]
        bnew = (b1 + b2) / 2.0
    smo.b = bnew
    smo.alphas[i1] = a1
    smo.alphas[i2] = a2
    smo.w = smo.X.T * np.multiply(smo.alphas, smo.y)
    for i in range(smo.m):
        if (smo.alphas[i] > 0) and (smo.alphas[i] < smo.C):
            smo.e_cache[i] = smo.X[i] * smo.w + smo.b - smo.y[i]
    return 1

def examine_example(i2, smo):
    y2 = smo.y[i2]
    alpha2 = smo.alphas[i2]
    if alpha2 > 0 and alpha2 <smo.C:
        E2 = smo.e_cache[i2]
    else:
        E2 = smo.X[i2] * smo.w + smo.b - smo.y[i2]
        smo.e_cache[i2] = E2
    r2 = E2 * y2
    if((r2 < -smo.tol) and (smo.alphas[i2] < smo.C)) or ((r2 > smo.tol) and (smo.alphas[i2] > 0)):
        # heuristic 1: find the max deltaE
        max_delta_E = 0
        i1 = -1
        for i in range(smo.m):
            if smo.alphas[i] > 0 and smo.alphas[i] < smo.C:
                if i == i2:
                    continue
                E1 = smo.e_cache[i]
                delta_E = abs(E1 - E2)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    i1 = i
        if i1 >= 0:
            if take_step(i1, i2, smo):
                return 1
        # heuristic 2: find the suitable i1 on border at random
        random_index = np.random.permutation(smo.m)
        for i in random_index:
            if smo.alphas[i] > 0 and smo.alphas[i] < smo.C:
                if i == i2:
                    continue
                if take_step(i, i2, smo):
                    return 1
        # heuristic 3: find the suitable i1 at random on all alphas
        random_index = np.random.permutation(smo.m)
        for i in random_index:
            if i == i2:
                continue
            if take_step(i1, i2, smo):
                return 1
    return 0

def main_routine(smo, max_iter=5):
    num_changed = 0
    examine_all = 1
    passes = 0
    while(passes <= max_iter):
        num_changed = 0
        if (examine_all == 1):
            for i2 in range(smo.m):
                num_changed += examine_example(i2, smo)
        else:
            for i2 in range(smo.m):
                if (smo.alphas[i2] > 0) and (smo.alphas[i2] < smo.C):
                    num_changed += examine_example(i2, smo)
        if (num_changed == 0):
            passes += 1
        if (examine_all == 1):
            examine_all = 0
        elif (num_changed == 0):
            examine_all = 1



data_mat = []
data_label = []
fr = open('testSetRBF')
for line in fr.readlines():
    line_arr = line.strip().split('\t')
    data_mat.append([float(element) for element in line_arr[:-1:]])
    data_label.append(float(line_arr[-1]))
smo = SMOStruct(data_mat, data_label, 100, 0.001, ('rbf', 0.1))
main_routine(smo)
error = 0.0
for i in range(smo.m):
    if smo.y[i] * (smo.X[i] * smo.w + smo.b) < 0:
        error += 1.0
print 'error rate is {}'.format(error/smo.m)
print 'b is {}'.format(smo.b)
print 'w is {}'.format(smo.w)