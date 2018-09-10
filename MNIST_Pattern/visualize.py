__author__ = 'Haohan Wang'

import numpy as np

from matplotlib import pyplot as plt

def loadTxt(filename):
    TR = []
    VAL = []
    TE = []
    for i in range(5):
        updateTest = True
        maxVal = 0
        text = [line.strip() for line in open('../results/MNIST_Pattern_Confound/'+ filename + '_' + str(i) + '.txt')]
        tr = []
        val = []
        te = []
        for line in text:
            if line.startswith('Epoch'):
                items = line.split()
                tr.append(float(items[8][:-1]))
                val.append(float(items[-1]))
                if len(val) == 0:
                    updateTest = True
                else:
                    if val[-1] > maxVal:
                        updateTest = True
                        maxVal = val[-1]
                    else:
                        updateTest = False
            if line.startswith('Best'):
                if updateTest:
                    te.append(float(line.split()[-1]))
                else:
                    te.append(te[-1])
        print te[-1]
        TR.append(tr)
        VAL.append(val)
        TE.append(te[:-1])
    TR = np.array(TR)
    VAL = np.array(VAL)
    TE = np.array(TE)

    return TR, VAL, TE

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

def plot(corr=0):
    tr1, val1, te1 = loadTxt('baseline_'+str(corr))
    tr2, val2, te2 = loadTxt('hex_'+str(corr))

    plot_mean_and_CI(np.mean(tr1, 0), np.mean(tr1, 0)-np.std(tr1,0), np.mean(tr1, 0)+np.std(tr1,0), color_mean='b--', color_shading='c')
    plot_mean_and_CI(np.mean(te1, 0), np.mean(te1, 0)-np.std(te1,0), np.mean(te1, 0)+np.std(te1,0), color_mean='b', color_shading='c')
    plot_mean_and_CI(np.mean(val1, 0), np.mean(val1, 0)-np.std(val1,0), np.mean(val1, 0)+np.std(val1,0), color_mean='b.', color_shading='c')

    plot_mean_and_CI(np.mean(tr2, 0), np.mean(tr2, 0)-np.std(tr2,0), np.mean(tr2, 0)+np.std(tr2,0), color_mean='r--', color_shading='m')
    plot_mean_and_CI(np.mean(te2, 0), np.mean(te2, 0)-np.std(te2,0), np.mean(te2, 0)+np.std(te2,0), color_mean='r', color_shading='m')
    plot_mean_and_CI(np.mean(val2, 0), np.mean(val2, 0)-np.std(val2,0), np.mean(val2, 0)+np.std(val2,0), color_mean='r.', color_shading='m')

    plt.legend(loc=4)
    plt.savefig('MNIST_Pattern_Confound_'+str(corr)+'.pdf')
    plt.clf()

if __name__ == '__main__':
    for i in range(3):
        plot(i)
