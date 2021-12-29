import matplotlib.pyplot as plt
import multiprocessing as mp
import math
import time
import numpy as np

# producing Julia sets is very similar to producing the Mandelbrot set, utilizing a similar equation...
# f(z) = z^2 - c. Each Julia set is identified by complex-valued c as a seed, and a complex value z is considered...
# to be in c's Julia set if f^n(z) is bounded for all n.

def julia(cr, ci, zr, zi, maxIters):
    iters = 0

    while math.sqrt(zr*zr + zi*zi) <= (1 + math.sqrt(1 + 4*math.sqrt(cr*cr + ci*ci)))/2 and iters != maxIters:
        iters = iters + 1

        zrTemp = zr
        zr = zr * zr - zi * zi + cr                         
        zi = 2 * zrTemp * zi + ci

    return iters

if __name__ == '__main__':

    start = time.time()

    minR = -2      # choose minimum real value to sample from
    maxR = 2      # choose maximum real value to sample from
    minI = -2      # smame for minimum imaginary...
    maxI = 2      # and maximum imaginary
    res = 800          # choose the resolution of the image; that is, how many numbers between minR/minI and maxR/maxI to evaluate
    maxIters = 200      # choose a limit for the number of iterations
    
    cr = 0.14
    ci = -0.9
    
    x, y = np.linspace(minR, maxR, res),  np.linspace(maxI, minI, res)      # let x and y be lists containing a res number of evenly spaced numbers from minR/I to maxR/I
    # maxI and minI are flipped so that when our information is transformed to a matrix, entries with a larger imainary part are at the top of the matrix...
    # while entries with a lesser imaginary part are at the bottom

    result = []     # initialize result as an empty list. This will contain the calulation of mandelbrot for every element of x and y

    with mp.Pool(mp.cpu_count()) as p:                                                  # divide the problem amongst all the logical processors in the machine                                                                      
        result.append(p.starmap(julia, [(cr, ci, i, j, maxIters) for j in y for i in x]))  # append the calculation of mandelbrot for eah entry in x (columns) and y (rows) to result

    z = np.array(result).reshape(res, res) # for each imaginary value...
    # for each real value, create an array of mandelbrot for each value and the provided maxIters
    # reshape it into a res X res matrix

    if cr == 0:
        if ci == 0:
            title = '0'
        else:
            title = str(ci) + 'i'
    else:
        if ci == 0:
            title = str(cr)
        elif ci < 0:
            title = str(cr) + str(ci) + 'i'
        else:
            title = str(cr) + '+' + str(ci) + 'i'

    plt.imshow(z, cmap = 'twilight_shifted', interpolation = 'bilinear', extent = [minR, maxR, minI, maxI]) # plot matrix z with bilinear interpolation
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('c=' + title)

    end = time.time()

    print('Execution time: ' +  str(end - start))

    plt.show() # output plot
