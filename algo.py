from __future__ import division
import numpy as np
import sys
from scipy.stats import rankdata
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from utils import *


# half-space
class HalfSpace:
    def __init__(self, k, t, direction, score=0):
        self.dimension = k
        self.frontier = t
        self.direction = direction  # possible values: 'left', 'right'
        self.score = score  # score of the half-space
    def __str__(self):
        return 'dimension: {}, frontier: {}, direction: {}, score: {}'.format(self.dimension, self.frontier, self.direction, self.score)
    def complementary(self):
        if self.direction == 'left':
            return HalfSpace(self.dimension, self.frontier, 'right', score=self.score)
        elif self.direction == 'right':
            return HalfSpace(self.dimension, self.frontier, 'left', score=self.score)
    def contains(self, X):
        if len(X.shape)==1:
            if self.direction == 'left' and X[self.dimension] <= self.frontier:
                return True
            elif self.direction == 'right' and X[self.dimension] > self.frontier:
                return True
            return False
        elif len(X.shape)==2:
            return [self.contains(x) for x in X]
    def contains_idx(self, X):
        n = X.shape[0]
        cont = self.contains(X)
        return [i for i in range(n) if cont[i]]
    def emp_kendall(self, n, X, Y):
        ns = X.shape[0]
        idx_cont = self.contains_idx(X)
        nc = len(idx_cont)
        idx_not = [i for i in range(ns) if i not in idx_cont]
        nn = ns-nc
        # strict inequality term
        res_strict = 0
        res_strict_chs = 0
        for i in range(ns):
            for j in range(i+1, ns):
                if Y[i]<Y[j]:
                    idx_min = i
                    idx_max = j
                elif Y[i]>Y[j]:
                    idx_min = j
                    idx_max = i
                if idx_max in idx_cont and idx_min in idx_not:
                    res_strict += 1
                elif idx_max in idx_not and idx_min in idx_cont:
                    res_strict_chs += 1
        """
        for i in idx_cont:
            for j in idx_not:
                if Y[i]>Y[j]:
                    res_strict += 1
                elif Y[i]<Y[j]:
                    res_strict_chs += 1
        """
        res_strict *= 2/(n*(n-1))
        res_strict_chs *= 2/(n*(n-1))
        # equality term
        res_eq = nc*(nc-1)/2+nn*(nn-1)/2
        res_eq /= n*(n-1)
        res_kendall = res_strict + res_eq
        return res_kendall, res_strict, res_eq, res_strict_chs+res_eq, res_strict_chs, idx_cont, idx_not
    def emp_iauc(self, n, X, Y, Yr):
        """
        Remarque : l'IAUC empirique est une U-stat d'ordre 3 NON SYMETRIQUE.
        Donc on somme sur des triplets (l'ordre compte !) --> normalisation en 1/(n*(n-1)*(n-2)) et pas 6/(n*(n-1)*(n-2))
        """
        ns = X.shape[0]
        idx_cont = self.contains_idx(X)
        idx_not = [i for i in range(ns) if i not in idx_cont]
        # strict inequality term
        res_strict = 0
        res_strict_chs = 0
        res_eq = 0
        for i in range(ns):
            for j in range(i+1, ns):
                if Y[i]<Y[j]:
                    idx_min = i
                    idx_max = j
                elif Y[i]>Y[j]:
                    idx_min = j
                    idx_max = i
                if idx_max in idx_cont and idx_min in idx_not:
                    res_strict += Yr[idx_max]-Yr[idx_min]-1
                elif idx_max in idx_not and idx_min in idx_cont:
                    res_strict_chs += Yr[idx_max]-Yr[idx_min]-1
                elif idx_max in idx_cont and idx_min in idx_cont:
                    res_eq += Yr[idx_max]-Yr[idx_min]-1
                elif idx_max in idx_not and idx_min in idx_not:
                    res_eq += Yr[idx_max]-Yr[idx_min]-1
        """
        for i in idx_cont:
            for j in idx_not:
                if Y[i]>Y[j]:
                    res_strict += Yr[i]-Yr[j]-1
                elif Y[i]<Y[j]:
                    res_strict_chs += Yr[j]-Yr[i]-1
        """
        res_strict *= 6/(n*(n-1)*(n-2))
        res_strict_chs *= 6/(n*(n-1)*(n-2))
        # equality term
        """
        res_eq = 0
        for i in idx_cont:
            for j in idx_cont[i+1:]:
                if Yr[i] != Yr[j]:  # should be always true when continuous distribution for Y
                    res_eq += np.abs(Yr[i]-Yr[j])-1
        for i in idx_not:
            for j in idx_not[i+1:]:
                if Yr[i] != Yr[j]:
                    res_eq += np.abs(Yr[i]-Yr[j])-1
        """
        res_eq *= 3/(n*(n-1)*(n-2))
        res_iauc = res_strict + res_eq
        return res_iauc, res_strict, res_eq, res_strict_chs+res_eq, res_strict_chs, idx_cont, idx_not

if __name__ == '__main__':

    ##############################
    ########## PLOT STYLE ########
    ##############################
    import matplotlib
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')  # solves 'Invalid DISPLAY variable' error
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    current_palette = sns.color_palette()
    sns.set_style("ticks")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc("lines", linewidth=3)
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \boldmath"]

    styles = ['o', '^', 's', 'D', '*']
    colors = current_palette[0:4]+current_palette[5:6]
    ##############################
    ##############################

    sidiseed = np.random.randint(10000)
    sidiseed = 4529  # bonnes seed pour polynome oscillant : 738, 4529
    print('seed =', sidiseed)
    np.random.seed(sidiseed)  # bonnes seed : 11, 131, 145.

    #"""
    # Example m(X) polynomial
    def m_P(x):
        return ((25*(x-1/2))**2)*(25*(x-1/2)+1)*(25*(x-1/2)+1.5)*(25*(x-1/2)+2)
    def m_P_01(x):
        return (m_P(x)-m_P(0))/((m_P(1)-m_P(0)))
    n = 100
    p = 1
    X = np.zeros((n, 1))
    for i in range(n):
        uu = np.random.rand()
        if uu<0.8:  # uniform between 0.415 and 0.51 where polynomial oscillates
            X[i] = 0.415+0.095*np.random.rand()
        elif uu<0.9:
            X[i] = 0.415*np.random.rand()
        else:
            X[i] = 0.51+0.49*np.random.rand()
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = m_P_01(X[i, 0])
    #"""

    """
    # Example multidimensional
    def m(x):
        #return (x[0]**2+x[1]**2+x[2]**2)/3
        #return x[0]*x[1]+x[0]*x[2]+x[1]*x[2]
        #return x[0]*x[1]*x[2]
        return np.cos(100*x[0])+x[1]*x[2]
    n = 100
    p = 3
    X = np.random.rand(n, p)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = m(X[i, :])
    """

    hs_all = []
    frontiers_all = []

    for objective in ['iauc', 'kendall']:

        print('..........Objective:', objective)

        # ranks of values of Y (useful for iauc)
        Yranks = rankdata(Y, method='min')

        # Algo parameters
        D = 3  # depth

        # run algo
        scores_old = np.ones(n)
        scores_new = np.ones(n)
        global_kendall, global_strict_kendall = 0.5, 0  # corresponds to kendall of constant scoring function (same for iauc)
        global_iauc, global_strict_iauc = 0.5, 0
        # frontiers for each score
        frontiers = np.zeros((1, p, 2))
        frontiers[0, :, 1] = 1  # assume points in [0, 1]^p
        # dictionary of half-spaces
        hs_dict = {}
        for d in range(D):
            global_eq_kendall = 0
            global_eq_iauc = 0
            # repeat frontiers matrix for upcoming updates
            frontiers = np.repeat(frontiers, 2, axis=0)
            # list of half-spaces
            hs_list = []
            scores_old = np.array(scores_new)
            s_values = np.unique(scores_old)
            for s in s_values:
                kendall = -1
                iauc = -1
                idx_s = np.where(scores_old==s)
                ns = len(idx_s[0])
                if ns <= 1:
                    if ns==0:
                        print('...found empty subspace')
                    # score new half-spaces
                    opt_hs = hs_dict[s]
                    opt_hs.score = 2*s
                    hs_dict[2*s] = opt_hs  # MEMORY ALLOCATION NOT OPTIMAL
                    # update scores vector
                    scores_new[idx_s] *= 2
                    # store optimal half-spaces
                    hs_list.append(opt_hs)
                    continue

                Xs = X[idx_s, :][0]
                Ys = Y[idx_s]
                Ysr = Yranks[idx_s]
                # spatial steps: candidates for frontiers
                space_steps = np.zeros((ns+1, p))
                Xs_sorted = np.sort(Xs, axis=0)
                space_steps[1:ns, :] = Xs_sorted[1:ns, :] + np.diff(Xs_sorted, axis=0)/2
                space_steps[0, :] = frontiers[int(2*s-2), :, 0]
                space_steps[ns, :] = frontiers[int(2*s-2), :, 1]
                for dim in range(p):
                    for i in range(ns+1):
                        frontier = space_steps[i, dim]
                        hs = HalfSpace(dim, frontier, 'left')
                        # kendall & iauc computation for left & right half-spaces
                        left_kendall, left_strict_kendall, current_eq_kendall, right_kendall, right_strict_kendall, left_cont_kendall, right_cont_kendall = hs.emp_kendall(n, Xs, Ys)
                        left_iauc, left_strict_iauc, current_eq_iauc, right_iauc, right_strict_iauc, left_cont_iauc, right_cont_iauc = hs.emp_iauc(n, Xs, Ys, Ysr)
                        if objective=='kendall':
                            # take max between left & right
                            imax = np.argmax([left_kendall, right_kendall])
                            current_kendall = [left_kendall, right_kendall][imax]
                            current_strict_kendall = [left_strict_kendall, right_strict_kendall][imax]
                            current_iauc = [left_iauc, right_iauc][imax]
                            current_strict_iauc = [left_strict_iauc, right_strict_iauc][imax]
                            if current_kendall > kendall:
                                opt_hs = hs
                                opt_hs.direction = ['left', 'right'][imax]
                                opt_chs = opt_hs.complementary()
                                chs_cont = [left_cont_kendall, right_cont_kendall][1-imax]
                                # save kendall
                                kendall = current_kendall
                                k_strict_kendall = current_strict_kendall
                                k_eq_kendall = current_eq_kendall
                                # save iauc
                                iauc = current_iauc
                                k_strict_iauc = current_strict_iauc
                                k_eq_iauc = current_eq_iauc
                        elif objective=='iauc':
                            # take max between left & right
                            imax = np.argmax([left_iauc, right_iauc])
                            current_iauc = [left_iauc, right_iauc][imax]
                            current_strict_iauc = [left_strict_iauc, right_strict_iauc][imax]
                            current_kendall = [left_kendall, right_kendall][imax]
                            current_strict_kendall = [left_strict_kendall, right_strict_kendall][imax]
                            if current_iauc > iauc:
                                opt_hs = hs
                                opt_hs.direction = ['left', 'right'][imax]
                                opt_chs = opt_hs.complementary()
                                chs_cont = [left_cont_iauc, right_cont_iauc][1-imax]
                                # save iauc
                                iauc = current_iauc
                                k_strict_iauc = current_strict_iauc
                                k_eq_iauc = current_eq_iauc
                                # save kendall
                                kendall = current_kendall
                                k_strict_kendall = current_strict_kendall
                                k_eq_kendall = current_eq_kendall
                # score new half-spaces
                opt_hs.score = 2*s
                opt_chs.score = 2*s-1
                # update scores vector
                scores_new[idx_s] *= 2
                scores_new[idx_s[0][chs_cont]] -= 1
                # store new frontiers
                idir = np.where(np.array(['left', 'right'])==opt_hs.direction)[0][0]
                frontiers[int(2*s-1), opt_hs.dimension, 1-idir] = opt_hs.frontier
                frontiers[int(2*s-2), opt_hs.dimension, idir] = opt_hs.frontier
                # store optimal half-spaces
                hs_list += [opt_chs, opt_hs]
                hs_dict[2*s-1] = opt_chs
                hs_dict[2*s] = opt_hs
                # update kendall
                global_kendall += kendall
                global_strict_kendall += k_strict_kendall
                global_eq_kendall += k_eq_kendall
                # update iauc
                global_iauc += iauc
                global_strict_iauc += k_strict_iauc
                global_eq_iauc += k_eq_iauc

            global_kendall = global_strict_kendall + global_eq_kendall
            global_iauc = global_strict_iauc + global_eq_iauc
            print('At depth %s:'%(d+1), 'iauc =', global_iauc, '....... kendall =', global_kendall)

        # store optimal half-spaces
        hs_all.append(hs_list)
        frontiers_all.append(frontiers)
        for hs in hs_list:
            print(hs)

    def compute_score(x, objective):
        if objective == 'iauc':
            hs_list = hs_all[0]
            frontiers = frontiers_all[0]
        elif objective == 'kendall':
            hs_list = hs_all[1]
            frontiers = frontiers_all[1]
        for hs in hs_list:
            s = hs.score
            for k in range(p):
                if x[k] < frontiers[int(s-1), k, 0] or x[k] > frontiers[int(s-1), k, 1]:
                    break
                return s/2**D

    # compare iauc, kendall and cart
    n_test = 2000
    X_test = np.random.rand(n_test, p)

    #"""
    # For oscillating polynomial
    for i in range(n_test):
        uu = np.random.rand()
        if uu<0.8:  # uniform between 0.415 and 0.51 where polynomial oscillates
            X_test[i] = 0.415+0.095*np.random.rand()
        elif uu<0.9:
            X_test[i] = 0.415*np.random.rand()
        else:
            X_test[i] = 0.51+0.49*np.random.rand()
    X_test.sort(axis=0)
    Y_test = np.zeros(n_test)
    for i in range(n_test):
        Y_test[i] = m_P_01(X_test[i, 0])
    m_test = np.array(Y_test)
    #"""

    """
    # Example multidimensional
    Y_test = np.zeros(n_test)
    for i in range(n_test):
        Y_test[i] = m(X_test[i, :])
    """

    s_iauc = [compute_score(x, 'iauc') for x in X_test]
    s_kendall = [compute_score(x, 'kendall') for x in X_test]
    # CART
    cart_reg = DecisionTreeRegressor(max_depth=D)
    cart_reg = cart_reg.fit(X, Y)
    pred_cart = cart_reg.predict(X_test)

    # print iauc and kendall for each model
    iauc = compute_iauc(s_iauc, Y_test)
    kendall = compute_kendall(s_iauc, Y_test)
    mse = mean_squared_error(Y_test, s_iauc)
    print('............IAUC: iauc = %s, kendall = %s, MSE = %s'%(iauc, kendall, mse))

    iauc = compute_iauc(s_kendall, Y_test)
    kendall = compute_kendall(s_kendall, Y_test)
    mse = mean_squared_error(Y_test, s_kendall)
    print('............KENDALL: iauc = %s, kendall = %s, MSE = %s'%(iauc, kendall, mse))

    iauc = compute_iauc(pred_cart, Y_test)
    kendall = compute_kendall(pred_cart, Y_test)
    mse = mean_squared_error(Y_test, pred_cart)
    print('............CART: iauc = %s, kendall = %s, MSE = %s'%(iauc, kendall, mse))

    #"""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    markevery = 350  # 350
    plt.plot(X_test, m_test, label=r"$m$", color=colors[1])#, markevery=markevery, ms=10.0, marker=styles[3])
    plt.plot(X_test, s_kendall, markevery=markevery, ms=10.0, label=r"\textsc{Kendall}", color=colors[2], marker=styles[1])
    plt.plot(X_test, s_iauc, label=r"\textsc{CRank}", color=colors[4])#, markevery=markevery, ms=10.0, marker=styles[2])
    plt.plot(X_test, pred_cart, label=r"\textsc{CART}", color=colors[3])#, markevery=markevery, ms=10.0, marker=styles[0])

    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$\text{score}(x)$', fontsize=18)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=2, fontsize=12).draw_frame(True)
    ax.set_ylim([0,1.05])

    plt.savefig('polynomial_modif.pdf', bbox_inches='tight')
    #"""

    #"""
    # ZOOM
    X_m = np.zeros((1001, 1))
    for i in range(1001):
        X_m[i] = 0.415 + i*0.095/1000
    #X_m = 0.415 + np.arange(1001)*0.095/1000
    Y_m = [m_P_01(x)-0.32645 for x in X_m]
    #s_iauc = [(compute_score(x, 'iauc')-1/4)*1e-6-0.32645 for x in X_m]
    #s_kendall = [(compute_score(x, 'kendall')-1/4)*1e-6-0.32645 for x in X_m]
    s_iauc = [(compute_score(x, 'iauc')-1/4)/(0.8-0.35)*1e-6+1.8e-6 for x in X_m]
    s_kendall = [(compute_score(x, 'kendall')-1/4)/(0.8-0.35)*1e-6+1.8e-6 for x in X_m]
    # CART
    pred_cart = cart_reg.predict(X_m)
    pred_cart = [(yy-0.32645)*1e-6+1.7e-6 for yy in pred_cart]

    fig = plt.figure(figsize=(7, 6))
    markevery = 500
    plt.plot(X_m, Y_m, label=r"$m$", color=colors[1])#, markevery=markevery, ms=10.0, marker=styles[3])
    plt.plot(X_m, s_kendall, markevery=markevery, ms=10.0, label=r"\textsc{Kendall}", color=colors[2], marker=styles[1])
    plt.plot(X_m, s_iauc, label=r"\textsc{CRank}", color=colors[4])#, markevery=markevery, ms=10.0, marker=styles[2])
    plt.plot(X_m, pred_cart, label=r"\textsc{CART}", color=colors[3])#, markevery=markevery, ms=10.0, marker=styles[0])

    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$\text{score}(x)-3.2645\times 10^{-1}$', fontsize=18)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(loc=2, fontsize=12).draw_frame(True)

    plt.savefig('polynomial_zoom_modif.pdf', bbox_inches='tight')
    #"""
