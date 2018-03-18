from init import *
import ga_method as ga
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    num_args = len(sys.argv)
    f = "test01.txt"
    draw = True
    if num_args > 2:
        f = sys.argv[1]
        draw = (sys.argv[2] == "True")
    tourManager = init(f)
    gs = ga.greedy_sol(tourManager)
    print ""
    if (draw):
        stats = ga.ga_sol(tourManager, 20, 50)
        best, aver = zip(*stats)
        x = range(len(stats))   
        plt.plot(x, [gs.score()] * len(stats), 'g-')
        plt.plot(x, best, 'b-')
        plt.plot(x, aver, 'r-')
        plt.show()

if __name__=='__main__':
    main()
