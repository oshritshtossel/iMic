import os
import pickle
from collections import OrderedDict, defaultdict
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm


def proccess_pickle(path, dd=None):
    f = open(path, "rb")
    d = pickle.load(f)
    f.close()

    baseline = dict()
    for f, k in d.keys():
        if f == 1:
            baseline[k] = d[(f, k)]

    # mm = mean(baseline[k] for k in baseline.keys())
    # for f, k in d.keys():
    #     if f ==1.:
    #         d[(f, k)] = 0
    #     else:
    #         d[(f, k)] -= mm
    for f, k in d.keys():
        d[(f, k)] -= baseline[k]

    if dd is None:
        D = defaultdict(OrderedDict)
    else:
        D = dd
    for f, k in d.keys():
        if dd is None:
            D[k][f] = d[(f, k)]
        else:
            D[k][f] += d[(f, k)]
    return D


if __name__ == '__main__':
    def get_cmap(n, name='viridis'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


    dir_ = "sensetivity_results/CD/CNN2"
    cmap = get_cmap(18)

    for i, p in enumerate(os.listdir(dir_)):
        p = dir_ + "/" + p
        if i == 0:
            dd = proccess_pickle(p)
        else:
            dd = proccess_pickle(p, dd)

    for k, ind in zip(dd.keys(), range(len(dd.keys()))):
        plt.scatter([-30, -20, -10, 0, 10, 20, 30], [j / (i + 1) for j in dd[k].values()], label=k.replace("_"," ").capitalize(), c=cmap(ind))

    plt.legend(bbox_to_anchor=(1.0, 1.0),prop={"family":"Times New Roman"})
    plt.xlabel("Percent of change", family='Times New Roman',fontsize=15)
    plt.ylabel("Test AUC change", family='Times New Roman',fontsize=15)
    plt.show()
