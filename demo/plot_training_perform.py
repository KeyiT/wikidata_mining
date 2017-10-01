import json
import matplotlib.pyplot as plt
import numpy as np


def plot_rf_train_performance():
    src_data_file = "results/training_performance_rf.json"

    # load json data
    with open(src_data_file, "r") as f:
        jsondata = json.load(f)

    data = {}
    for tr in jsondata:
        nun_est = tr['n_estimators']
        if nun_est not in data:
            data.update(
                {nun_est: {'max_depth': [], 'mean': []}}
            )

        data[nun_est]['max_depth'].append(tr['max_depth'])
        data[nun_est]['mean'].append(tr['mean'])

    plt.figure()

    init = 0.1
    step = 20
    unit = (1 - init) / (90 - 10) * step
    k = 0
    for ne in xrange(10, 91, step):
        plt.plot(data[ne]['max_depth'],
                 data[ne]['mean'], c='black', label="# DTs: " + str(ne), alpha=(init + unit * k))
        k += 1

    plt.legend(loc=4)
    plt.ylabel('F-Measure')
    plt.xlabel('Max Depth')
    plt.savefig("results/rf_depth_numDTs_accuracy.png")


def plot_lr_train_performance():
    src_data_file = "results/train_performance_lr.json"

    # load json data
    with open(src_data_file, "r") as f:
        jsondata = json.load(f)

    data = {'C': [], 'mean': []}

    for tr in jsondata:
        data['C'].append(tr['C'])
        data['mean'].append(tr['mean'])

    plt.figure()

    plt.plot(data['C'], data['mean'], c='black')

    plt.legend(loc=4)
    plt.ylabel('F-Measure')
    plt.xlabel('Cs')
    plt.savefig("results/lr_Cs_accuracy.png")


plot_lr_train_performance()