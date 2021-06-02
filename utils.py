import json
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd, mkdir
from os.path import join, isdir

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def plot(exp_name, plot_name, value_dict, scatter=False):
    c = getcwd()
    co = join(c, 'output')
    coe = join(co, exp_name)
    if not isdir(coe):
        if not isdir(co):
            mkdir(co)
        mkdir(coe)
    save_path = join(coe, f'{plot_name}.png')
    for label, values in value_dict.items():
        if scatter:
            iters = [i for i, v in enumerate(values) if v >= 0]
            values = [v for v in values if v >= 0]
            plt.scatter(iters, values, label=label)
        else:
            plt.plot(values, label=label)
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(save_path)
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window
    