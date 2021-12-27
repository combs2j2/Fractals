import numpy as np # installed with matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

def mandelbrot(cr, ci, maxIters):
    iters, zr, zi = 0, 0, 0

    while iters < maxIters and zr * zr + zi * zi <= 4:
        iters = iters + 1
        zrTemp = zr

        zr = zr * zr - zi * zi + cr
        zi = 2 * zrTemp * zi + ci

    return iters    
    

if __name__ == '__main__':

    minR = -0.17     # choose minimum real value to sample from
    maxR = -0.15      # choose maximum real value to sample from
    minI = 1.025      # smame for minimum imaginary...
    maxI = 1.045      # and maximum imaginary
    res = 1080          # choose the resolution of the image; that is, how many numbers between minR/minI and maxR/maxI to evaluate
    maxIters = 200      # choose a limit for the number of iterations 
    
    x, y = np.linspace(minR, maxR, res),  np.linspace(maxI, minI, res)

    result = []

    with Pool(3) as p:                                                              
        result.append(p.starmap(mandelbrot, [(i, j, maxIters) for j in y for i in x]))

    z = np.array(result).reshape(res, res) # for each imaginary value...
    # for each real value, create an array of mandelbrot for each value and the provided maxIters
    # reshape it into a res X res matrix

    plt.imshow(z, cmap = 'twilight_shifted', interpolation = 'bilinear', extent = [minR, maxR, minI, maxI]) # plot matrix z with bilinear interpolation
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')

    plt.show() # output plot
