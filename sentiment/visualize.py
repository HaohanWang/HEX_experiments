__author__ = 'Haohan Wang'

import numpy as np

from matplotlib import pyplot as plt

import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)


plt.style.use('bmh')

def loadTxt(filename):
    print 'loading', filename
    TR = []
    VAL = []
    TE = []
    for i in range(1, 6):
        updateTest = True
        maxVal = 0
        text = [line.strip() for line in open('../results/sentiment/'+ filename + '_' + str(i) + '.txt')]
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

def loadTxtNew(filename):
    print 'loading', filename
    TR = []
    VAL = []
    TE = []
    for i in range(1, 6):
        updateTest = True
        maxVal = 0
        text = [line.strip() for line in open('../results/sentiment/'+ filename + '_' + str(i) + '.txt')]
        tr = []
        val = []
        te = []
        startUpdate = False
        for line in text:
            if line.startswith('Start'):
                startUpdate = True
            if startUpdate:
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
                            te.append(te[-1])
                if line.startswith('Best'):
                    if updateTest:
                        te.append(float(line.split()[-1]))

        TR.append(tr)
        VAL.append(val)
        TE.append(te[:100])
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
    plt.style.use('bmh')
    tr1, val1, te1 = loadTxt('baseline_'+str(corr))
    tr2, val2, te2 = loadTxt('vanilla_'+str(corr))
    tr3, val3, te3 = loadTxt('mlp_'+str(corr))
    tr4, val4, te4 = loadTxt('adv_'+str(corr))
    tr0, val0, te0 = loadTxt('hex_'+str(corr))
    tr5, val5, te5 = loadTxtNew('pre_'+str(corr))
    tr6, val6, te6 = loadTxtNew('info_'+str(corr))

    # plot_mean_and_CI(np.mean(tr1, 0), np.mean(tr1, 0)-np.std(tr1,0)/5.0, np.mean(tr1, 0)+np.std(tr1,0)/5.0, color_mean='g--', color_shading='g')
    plot_mean_and_CI(np.mean(te1, 0), np.mean(te1, 0)-np.std(te1,0), np.mean(te1, 0)+np.std(te1,0), color_mean='g', color_shading='g')
    # plot_mean_and_CI(np.mean(val1, 0), np.mean(val1, 0)-np.std(val1,0)/5.0, np.mean(val1, 0)+np.std(val1,0)/5.0, color_mean='g.', color_shading='g')

    # plot_mean_and_CI(np.mean(tr2, 0), np.mean(tr2, 0)-np.std(tr2,0)/5.0, np.mean(tr2, 0)+np.std(tr2,0)/5.0, color_mean='b--', color_shading='b')
    plot_mean_and_CI(np.mean(te2, 0), np.mean(te2, 0)-np.std(te2,0), np.mean(te2, 0)+np.std(te2,0), color_mean='b', color_shading='b')
    # plot_mean_and_CI(np.mean(val2, 0), np.mean(val2, 0)-np.std(val2,0)/5.0, np.mean(val2, 0)+np.std(val2,0)/5.0, color_mean='b.', color_shading='b')

    # plot_mean_and_CI(np.mean(tr3, 0), np.mean(tr3, 0)-np.std(tr3,0)/5.0, np.mean(tr3, 0)+np.std(tr3,0)/5.0, color_mean='c--', color_shading='c')
    plot_mean_and_CI(np.mean(te3, 0), np.mean(te3, 0)-np.std(te3,0), np.mean(te3, 0)+np.std(te3,0), color_mean='c', color_shading='c')
    # plot_mean_and_CI(np.mean(val3, 0), np.mean(val3, 0)-np.std(val3,0)/5.0, np.mean(val3, 0)+np.std(val3,0)/5.0, color_mean='c.', color_shading='c')

    # plot_mean_and_CI(np.mean(tr4, 0), np.mean(tr4, 0)-np.std(tr4,0)/5.0, np.mean(tr4, 0)+np.std(tr4,0)/5.0, color_mean='m--', color_shading='m')
    plot_mean_and_CI(np.mean(te4, 0), np.mean(te4, 0)-np.std(te4,0), np.mean(te4, 0)+np.std(te4,0), color_mean='m', color_shading='m')
    # plot_mean_and_CI(np.mean(val4, 0), np.mean(val4, 0)-np.std(val4,0)/5.0, np.mean(val4, 0)+np.std(val4,0)/5.0, color_mean='m.', color_shading='m')

    # plot_mean_and_CI(np.mean(tr0, 0), np.mean(tr0, 0)-np.std(tr0,0)/5.0, np.mean(tr0, 0)+np.std(tr0,0)/5.0, color_mean='r--', color_shading='r')
    plot_mean_and_CI(np.mean(te0, 0), np.mean(te0, 0)-np.std(te0,0), np.mean(te0, 0)+np.std(te0,0), color_mean='r', color_shading='r')
    # plot_mean_and_CI(np.mean(val0, 0), np.mean(val0, 0)-np.std(val0,0)/5.0, np.mean(val0, 0)+np.std(val0,0)/5.0, color_mean='r.', color_shading='r')

    # plot_mean_and_CI(np.mean(tr5, 0), np.mean(tr5, 0)-np.std(tr5,0)/5.0, np.mean(tr5, 0)+np.std(tr5,0)/5.0, color_mean='y--', color_shading='y')
    plot_mean_and_CI(np.mean(te5, 0), np.mean(te5, 0)-np.std(te5,0), np.mean(te5, 0)+np.std(te5,0), color_mean='y', color_shading='y')
    # plot_mean_and_CI(np.mean(val5, 0), np.mean(val5, 0)-np.std(val5,0)/5.0, np.mean(val5, 0)+np.std(val5,0)/5.0, color_mean='y.', color_shading='y')

    # plot_mean_and_CI(np.mean(tr6, 0), np.mean(tr6, 0)-np.std(tr6,0)/5.0, np.mean(tr6, 0)+np.std(tr6,0)/5.0, color_mean='k--', color_shading='k')
    plot_mean_and_CI(np.mean(te6, 0), np.mean(te6, 0)-np.std(te6,0), np.mean(te6, 0)+np.std(te6,0), color_mean='k', color_shading='k')
    # plot_mean_and_CI(np.mean(val6, 0), np.mean(val6, 0)-np.std(val6,0)/5.0, np.mean(val6, 0)+np.std(val6,0)/5.0, color_mean='k.', color_shading='k')

    plt.legend(loc=4)
    plt.ylim(0,1.05)
    plt.savefig('sentiment_'+str(corr)+'.pdf')
    plt.clf()

def plotLegend():
    plt.style.use('bmh')
    methodsName = ['Baseline', 'Ablation M', 'Ablation N', 'Adv', 'HEX', 'DANN', 'InfoDrop']
    colors = ['g', 'c', 'b', 'm', 'r', 'y', 'k']

    fig = plt.figure(dpi=350, figsize=(20, 1))
    ax = fig.add_axes([0, 0, 0.001, 0.001])
    for i in range(len(colors)):
        ax.plot(xrange(10), xrange(10), label=methodsName[i], color=colors[i])
    plt.legend(loc="upper center", bbox_to_anchor=(500, 800), ncol=7)
    plt.savefig('legend.pdf')


def resultPlot():
    boxColors = ['darkkhaki', 'royalblue']

    fig = plt.figure(dpi=350, figsize=(25, 9))
    axs = [0 for i in range(10)]

    newFiles = ['pre', 'info']

    fileNames = ['baseline',  'mlp', 'vanilla', 'adv', 'hex', 'pre', 'info']
    labelNames = ['B', 'M', 'N', 'A', 'H', 'D', 'I']

    plt.style.use('bmh')

    for i in range(10):
        if i < 5:
            m = 1
            z = i%5
        else:
            m = 0
            z = i%5
        axs[i] = fig.add_axes([0.075+z*0.18, 0.1+m*0.45, 0.16, 0.35])

        ts = []
        for k in range(len(fileNames)):
            if fileNames[k] in newFiles:
                tr, val, te = loadTxtNew(fileNames[k]+'_'+str(i))
            else:
                tr, val, te = loadTxt(fileNames[k]+'_'+str(i))
            ts.append(te[:,-1])

        # m1 = np.mean(r1)
        # s1 = np.std(r1)
        # m2 = np.mean(r2)
        # s2 = np.std(r2)

        # axs[c].errorbar(x=[0, 1], y=[m1, m2], yerr=[s1, s2])

        axs[i].boxplot(ts, positions=[j for j in range(len(fileNames))], widths=[0.5 for j in range(len(fileNames))])
        # axs[c].boxplot(r2, positions=[1])

        axs[i].set_xlim(-0.5, len(fileNames)-0.5)
        axs[i].set_ylim(0.0, 1.1)

        if i == 0 or i == 5:
            axs[i].set_ylabel('Accuracy')

        axs[i].set_xticklabels(labelNames)
        # if c1 == 0:
        # axs[c].set_xticks([0, 1], ['NN', 'HEX-NN'])
        # else:
        #     axs[c].get_xaxis().set_visible(False)

        axs[i].title.set_text(r'$\rho$: '+str(i/float(10)))

    # plt.legend(loc="upper center", bbox_to_anchor=(1, 1), fancybox=True, ncol=2)
    plt.savefig('fig.pdf', dpi=350, format='pdf')


if __name__ == '__main__':
    plotLegend()
    # for i in range(10):
    #     plot(i)
    # resultPlot()