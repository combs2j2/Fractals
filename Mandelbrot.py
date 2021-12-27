import numpy as np # installed with matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

# the mandelbrot set creates what is considered one of the most elegent visualizations in mathematics
# the rules of the mandelbrot set are relatively simple; consider the recursive equation z(n+1) = z(n)^2 + c where z(i) and c are complex numbers and z(0) = 0
# if for all natural numbers n z(n) is bounded (that is, there are finite values that the magnitude of z(n) never exceeds), then our chosen c is in the set
# otherwise, z(n) diverges for some n, and our chosen c is not in the set

# for example, c = -1 is in the mandelbrot set because z(1) = z(0)^2 - 1 = 0 - 1 = -1, z(2) = z(1)^2 - 1 = 1 - 1 = 0, z(3) = z(2)^2 - 1 = -1,...
# clearly, z(n) is always either -1 or 0, so the magitude of z(n) is bounded by the range [0, 1] and is therefore in the set.

# however, c = 1 is not in the mandelbrot set because z(1) = z(0)^2 + 1 = 0 + 1 = 1, z(2) = z(1)^2 + 1 = 1 + 1 = 2, z(3) = z(2)^2 + 1 = 4 + 1 = 5,...
# as n increases, the magnitude of z(n) increaseswithout bound, so c = 1 is not in the set

# visualization of the mandelbrot set takes every complex number c, and assigns a color to how quickly it diverges (specifically how many iterations it takes for the magnitude of z(n) to exceed 2)
# since computers have limited memory, we assign a maximum number of iterations such that when our number of iterations reaches that limit, we "throw our hands up" and assume our z(n) are bounded
# the higher our limit is, the closer we get to the actual mandelbrot set

def mandelbrot(cr, ci, maxIters):           # takes a complex number c (cr and ci representing real and imaginary parts, respectively) and a limit for the number of iterations (maxIters)
    iters, zr, zi = 0, 0, 0                 # initialize iters at 0, as well as complex number z (broken up as zr and zi) as 0 by definition

    while iters < maxIters and zr * zr + zi * zi <= 4:      # end when iters reaches maxIters, or when the magnitude squared of z exceeds 4 (also by mathematical definition)
        iters = iters + 1                                   # increment iters
        zrTemp = zr                                         # set a temporary holder for the original value of zr

        zr = zr * zr - zi * zi + cr                         # real part of z^2 + c (using rules for operations on complex numbers)
        zi = 2 * zrTemp * zi + ci                           # imaginary part of z^2 + c

    return iters                                            # return the number of iterations it takes for z to diverge (magnited squared exceeds 4) or maxIters (approximately representing boundedness)
    

if __name__ == '__main__':

    minR = -2      # choose minimum real value to sample from
    maxR = 2      # choose maximum real value to sample from
    minI = -2      # smame for minimum imaginary...
    maxI = 2      # and maximum imaginary
    res = 1080          # choose the resolution of the image; that is, how many numbers between minR/minI and maxR/maxI to evaluate
    maxIters = 200      # choose a limit for the number of iterations 
    
    x, y = np.linspace(minR, maxR, res),  np.linspace(maxI, minI, res)      # let x and y be lists containing a res number of evenly spaced numbers from minR/I to maxR/I
    # maxI and minI are flipped so that when our information is transformed to a matrix, entries with a larger imainary part are at the top of the matrix...
    # while entries with a lesser imaginary part are at the bottom

    result = []     # initialize result as an empty list. This will contain the calulation of mandelbrot for every element of x and y

    with Pool(3) as p:                                                                  # create 3 threads (3 cores divide problem among themselves)                                                      
        result.append(p.starmap(mandelbrot, [(i, j, maxIters) for j in y for i in x]))  # append the calculation of mandelbrot for eah entry in x (columns) and y (rows) to result

    z = np.array(result).reshape(res, res) # for each imaginary value...
    # for each real value, create an array of mandelbrot for each value and the provided maxIters
    # reshape it into a res X res matrix

    plt.imshow(z, cmap = 'twilight_shifted', interpolation = 'bilinear', extent = [minR, maxR, minI, maxI]) # plot matrix z with bilinear interpolation
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')

    plt.show() # output plot
