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
    src_data_file = "results/train_performance_rflr.json"

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
    plt.xlabel('L1-norm Regularization Weights')
    plt.savefig("results/rflr_cs_accuracy.png")


def plot_abdt_train_performance():
    src_data_file = "results/train_performance_abdt.json"

    # load json data
    with open(src_data_file, "r") as f:
        jsondata = json.load(f)

    data = {}
    max_num_est = 0
    min_num_est = 1E10
    for tr in jsondata:
        nun_est = tr['n_estimators']
        max_num_est = int(np.maximum(max_num_est, int(float(nun_est))))
        min_num_est = int(np.minimum(min_num_est, int(float(nun_est))))
        if nun_est not in data:
            data.update(
                {nun_est: {'max_depth': [], 'mean': []}}
            )

        data[nun_est]['max_depth'].append(tr['max_depth'])
        data[nun_est]['mean'].append(tr['mean'])

    plt.figure()

    init = 0.1
    step = 20
    unit = (1 - init) / (max_num_est - min_num_est) * step
    k = 0
    for ne in xrange(min_num_est, max_num_est, step):
        plt.plot(data[ne]['max_depth'],
                 data[ne]['mean'], c='black', label="# DTs: " + str(ne), alpha=(init + unit * k))
        k += 1

    plt.legend(loc=4)
    plt.ylabel('F-Measure')
    plt.xlabel('Max Depth')
    plt.savefig("results/abdt_depth_numDTs_accuracy.png")


def plot_gbdt_train_performance():
    src_data_file = "results/train_performance_gbdt.json"

    # load json data
    with open(src_data_file, "r") as f:
        jsondata = json.load(f)

    data = {}
    max_num_est = 0
    min_num_est = 1E10
    for tr in jsondata:
        nun_est = tr['n_estimators']
        max_num_est = int(np.maximum(max_num_est, int(float(nun_est))))
        min_num_est = int(np.minimum(min_num_est, int(float(nun_est))))
        if nun_est not in data:
            data.update(
                {nun_est: {'max_depth': [], 'mean': []}}
            )

        data[nun_est]['max_depth'].append(tr['max_depth'])
        data[nun_est]['mean'].append(tr['mean'])

    plt.figure()

    init = 0.1
    step = 20
    unit = (1 - init) / (max_num_est - min_num_est) * step
    k = 0
    for ne in xrange(min_num_est, max_num_est, step):
        plt.plot(data[ne]['max_depth'],
                 data[ne]['mean'], c='black', label="# DTs: " + str(ne), alpha=(init + unit * k))
        k += 1

    plt.legend(loc=4)
    plt.ylabel('F-Measure')
    plt.xlabel('Max Depth')
    plt.savefig("results/gbdt_depth_numDTs_accuracy.png")


plot_lr_train_performance()