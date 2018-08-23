__author__ = 'Haohan Wang'

from matplotlib import pyplot as plt

def loadTxt(filename):
    text = [line.strip() for line in open(filename + '.txt')]
    score = []
    for line in text:
        if line.startswith('Best'):
            score.append(float(line.split()[-1]))
    return score


def plot():
    s1 = loadTxt('vanilla2')
    s2 = loadTxt('hex_2')
    s3 = loadTxt('hex_10')
    s4 = loadTxt('hex_20')

    x = xrange(101)
    plt.plot(x, s1, label='vanilla', lw=2)
    plt.plot(x, s2, label='hex_2', lw=2)
    plt.plot(x, s3, label='hex_10', lw=2)
    plt.plot(x, s4, label='hex_20', lw=2)

    plt.legend(loc=4)
    plt.show()

if __name__ == '__main__':
    plot()
