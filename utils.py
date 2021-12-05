from __future__ import division
import numpy as np
import sys
from scipy.stats import rankdata


def compute_kendall(sX, Y):
    n = len(Y)
    s_values = np.unique(sX)
    res_strict = 0
    res_eq = 0
    for s in s_values:
        idx_s = np.where(sX==s)
        idx_cont = idx_s[0]
        nc = len(idx_cont)
        idx_not = [i for i in range(n) if i not in idx_cont]
        # strict inequality term
        for i in range(n):
            for j in range(i+1, n):
                if Y[i]<Y[j]:
                    idx_min = i
                    idx_max = j
                elif Y[i]>Y[j]:
                    idx_min = j
                    idx_max = i
                if idx_max in idx_cont and idx_min in idx_not:
                    res_strict += 1
        """
        for i in idx_cont:
            for j in idx_not:
                if Y[i]>Y[j]:
                    res_strict += 1
        """
        # equality term
        res_eq += nc*(nc-1)/2
    res_eq /= n*(n-1)
    res_strict *= 2/(n*(n-1))
    res_kendall = res_strict + res_eq
    return res_kendall

def compute_iauc(sX, Y):
    """
    Remarque : l'IAUC empirique est une U-stat d'ordre 3 NON SYMETRIQUE.
    Donc on somme sur des triplets (l'ordre compte !) --> normalisation en 1/(n*(n-1)*(n-2)) et pas 6/(n*(n-1)*(n-2))
    """
    Yr = rankdata(Y, method='min')
    n = len(Y)
    s_values = np.unique(sX)
    res_strict = 0
    res_eq = 0
    for s in s_values:
        idx_s = np.where(sX==s)
        idx_cont = idx_s[0]
        nc = len(idx_cont)
        idx_not = [i for i in range(n) if i not in idx_cont]
        # strict inequality term
        for i in range(n):
            for j in range(i+1, n):
                if Y[i]<Y[j]:
                    idx_min = i
                    idx_max = j
                elif Y[i]>Y[j]:
                    idx_min = j
                    idx_max = i
                if idx_max in idx_cont and idx_min in idx_not:
                    res_strict += Yr[idx_max]-Yr[idx_min]-1
                elif idx_max in idx_cont and idx_min in idx_cont:
                    res_eq += Yr[idx_max]-Yr[idx_min]-1
        """
        for i in idx_cont:
            for j in idx_not:
                if Y[i]>Y[j]:
                    res_strict += Yr[i]-Yr[j]-1
        # equality term
        for i in idx_cont:
            for j in idx_cont[i+1:]:
                if Yr[i] != Yr[j]:  # should be always true when continuous distribution for Y
                    res_eq += np.abs(Yr[i]-Yr[j])-1
        """
    res_strict *= 6/(n*(n-1)*(n-2))
    res_eq *= 3/(n*(n-1)*(n-2))
    res_iauc = res_strict + res_eq
    return res_iauc
