import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import test_YY
import numpy as np
import matplotlib.ticker as ticker

###
### Setup for pyplot
###

#This may be changed to whatever backends work with the system
#See
#https://stackoverflow.com/questions/7534453/matplotlib-does-not-show-my-drawings-although-i-call-pyplot-show
plt.switch_backend('TkAgg')

plt.rc('text', usetex=True)

#Turns interactive mode on
#This has the effect of allowing us to run code while a window is open, and to run plt.show() many times.
plt.ion() 
plt.show()


def play_setup(fig):
    ax3d = fig.add_subplot(211, projection = '3d')
    ax3d.set_xlabel('X Label')
    ax3d.set_ylabel('Y Label')
    ax3d.set_zlabel('Z Label')
    #fig2 = plt.figure()
    ax = fig.add_subplot(212)
    return ax, ax3d

def double_plot(fig, Bs, trs, negs):
    """
    Makes two scatter plots on fig: one of Bs vs. trs, one of Bs vs. negs.
    """
    fig.clear()
    axtop = fig.add_subplot(211)
    axbot = fig.add_subplot(212)

    Bmax = max(Bs)
    axtop.scatter((Bs/Bmax), trs, marker=',', color='blue', s=1)
    axtop.set_ylim(0, max(trs)*1.05)
    axtop.set_ylabel("K-means objective")
    
    axbot.scatter((Bs/Bmax), negs, marker=',', color='blue', s=1)
    axbot.set_ylim(0, max(negs)*1.05)
    axbot.set_ylabel("Norm of negative part")
    
    return axtop, axbot

def double_plot_file(fig, suffix, pltpath=True):
    """
    Makes a "double_plot" using the data saved with suffix given by the string suffix.
    if pltpath is true then it also plots the path of a homotopy
    All this assumes that the data is saved by test_YY.gen_path and test_YY.gen_many_mins.
    """
    As, Bs, trs, negs = test_YY.load_mins(suffix)
    axtop, axbot =  double_plot(fig, Bs, trs, negs)
    if pltpath:
        _, Bs_path, trs_path, negs_path = test_YY.load_path(suffix)
        print(Bs_path)
        print(trs_path)
        axtop.scatter(Bs_path/max(Bs), trs_path, marker=',', color='red', s=1)
        axbot.scatter(Bs_path/max(Bs), negs_path, marker=',', color='red', s=1)
    return axtop, axbot
                     

def single_plot_file(fig, suffix, pltpath=True, sdp=None, lloyd=None, spec=None):
    """
    Creates a single plot of the homotopy path against trs.
    The x axis is log(Bs/As).
    If floats sdp, lloyd, and/or spec are given, also plots these values on the last point of the homotopy
    """
    As, Bs, trs, negs = test_YY.load_mins(suffix)
    lba = np.log(Bs/As)
    fakeinf = max(lba[lba<np.inf])*1.1
    fakeneginf = min(lba[lba>-np.inf])*1.1
    lba[lba==np.inf] = fakeinf
    lba[lba==-np.inf] = fakeneginf
    fig.clear()
    ax = fig.add_subplot(111)
    ax.scatter(lba, trs, marker=',', color='blue', s=1)
    ax.set_ylim(0, max(trs)*1.05)
    ax.set_ylabel("K-means objective")
    ax.set_xlabel(r"$\log(\beta)$")

    ax.xaxis.set_major_locator(ticker.FixedLocator(([0,fakeinf])))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((["0", r"$\infty$"])))
    
    if pltpath:
        As_path, Bs_path, trs_path, negs_path = test_YY.load_path(suffix)
        lba_path = np.log(Bs_path/As_path)
        lba_path[lba_path==np.inf] = fakeinf
        lba_path[lba_path==-np.inf] = fakeneginf
        ax.scatter(lba_path, trs_path, marker='x', color='red', s=25)

    if sdp is not None:
        ax.scatter([fakeinf], [sdp], color='green', s=25, marker='+')
    if lloyd is not None:
        ax.scatter([fakeinf], [lloyd], color='green', s=25, marker='x')
    if spec is not None:
        ax.scatter([fakeinf], [spec], color='green', s=25, marker='o')

    fig.tight_layout()
    return ax

def plot_paths(fig, pblm_test):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_ylabel("K-means objective")
    ax.set_xlabel(r"$t$")

    for ts, Ys in pblm_test.paths:
        trs = [pblm_test.pblm.tr(Y) for Y in Ys]
        ax.scatter(ts, trs, marker='x', s=25)

    fig.tight_layout()
    return ax

def reset3d():
    ax3d.clear()
    ax3d.set_xlabel('X Label')
    ax3d.set_ylabel('Y Label')
    ax3d.set_zlabel('Z Label')



