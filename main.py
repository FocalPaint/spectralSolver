import colour
import numpy as np
import sys

from colour.plotting import *

from scipy.optimize import differential_evolution, basinhopping
from solver import *
from plotting import draw_primaries, plotColorMixes
from tools import generateT_MATRIX_RGB
from itertools import repeat
from settings import *

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)


def func(a):
    return 0.0
sd = np.repeat(0.0, numwaves)
# whiteSpectrum = np.repeat(1.0, numwaves)


def objectiveFunction(a):

    sds, tmat = extractDataFromParameter(a)

    # do we care about slope?
    # result = minimize_slopes(sds) * weight_minslope
    # result += minimize_slopes(tmat) * weight_minslope

    result = match_XYZ(sds[0], XYZ[0], tmat) ** 2.0 * weight_red
    result += match_XYZ(sds[1], XYZ[1], tmat) ** 2.0 * weight_green
    result += match_XYZ(sds[2], XYZ[2], tmat) ** 2.0 * weight_blue
    result += match_XYZ(np.repeat(1.0, numwaves), white_XYZ, tmat) ** 2.0 * weight_illuminant_white * 1000.
    #result += varianceWaves(a) * weight_variance
    #result += uniqueWaves(a) * weight_uniqueWaves
    
    # nudge b+y = green
    yellow = sds[0] + sds[1]
    purple = (sds[0] * 0.5) + sds[2]
    orange = sds[0] + (sds[1] * 0.5)
    # green = (sds[0] + sds[1] + (sds[2])) 
    # result += mix_test(sds[2], yellow, green, 0.5, tmat) ** 2.0 * weight_mixtest1

    result += mix_test_RGB(sds.sum(axis=0), sds.sum(axis=0), np.array([1., 1., 1.]), 0.5, tmat) ** 2.0 * weight_mixtest1 * 1000

    result += mix_test_RGB(sds[2], yellow, np.array([0.214, 1.0, 0.214]), 0.5, tmat) ** 2.0 * weight_mixtest1
    # nudge b+w towards desaturated cyan
    # cyan = sds[1] + sds[2] + (sds[0] * 0.5)
    result += mix_test_RGB(sds[2], np.repeat(1.0, numwaves), [0.214, 0.3185, 1.0], 0.5, tmat) ** 2.0 * weight_mixtest2
    #blue and green should be cyan
    result += mix_test_RGB(sds[2], sds[1], np.array([0.0, 1.0, 1.0]), 0.5, tmat) ** 2.0 * weight_mixtest2
    # nudge b+r should be purple
    
    result += mix_test_RGB(sds[0], purple, np.array([0.5, 0.0, 1.0]), 0.5, tmat) ** 2.0 * weight_mixtest3
    # orange
    
    result += mix_test_RGB(sds[0], orange, np.array([1.0, 0.5, 0.0]), 0.5, tmat) ** 2.0 * weight_mixtest3

    # white and white

    result += mix_test_RGB(sds.sum(axis=0), sds.sum(axis=0), np.array([1.0, 1.0, 1.0]), 0.5, tmat) ** 2.0 * weight_mixtest1 * 100.


    # red mix straight with white, try to avoid going yellowish
    # lightRed = (sds[1] * 0.2) + (sds[2] * 0.5) + sds[0]
    result += mix_test_RGB(sds[0], np.repeat(1.0, numwaves), np.array([1.0, 0.214, 0.214]), 0.5, tmat) ** 2.0 * weight_mixtest1 

    # blue and orange should be greyish
    result += mix_test_RGB(sds[2], orange, np.repeat(0.214, 3), 0.5, tmat) ** 2.0 * weight_mixtest1

    # purple and yellow should be greyish
    result += mix_test_RGB(sds[2] + sds[0], sds[0] + sds[1], np.repeat(0.214, 3), 0.5, tmat) ** 2.0 * weight_mixtest1

    # red and green should be greyish
    result += mix_test_RGB(sds[0], sds[1], np.repeat(0.214, 3), 0.5, tmat) ** 2.0 * weight_mixtest1 * 10

    # primaries should sum to ~one for each channel to help conserve energy or something
    result += ((np.sum(sds,axis=0) - MAX_REFLECTANCE) ** 2.0).sum() * weight_sum_to_one

    # each primary integral should be 1.0
    result += ((np.sum(sds,axis=1) - 1.0) ** 2.0).sum() * weight_sum_to_one

    # tmat center weight sum to one
    result += ((np.sum(tmat[1]) - 1.0) ** 2.0) * weight_sum_to_one


    return result


def objectiveFunctionSingle(a, targetXYZ, spectral_to_XYZ_m):
    # result = minimize_slope(a)
    result = match_XYZ(a, targetXYZ, spectral_to_XYZ_m) * 1000.
    return result

if __name__ == '__main__':
    spdBounds = (MIN_REFLECTANCE, MAX_REFLECTANCE)
    from itertools import repeat
    bounds = (tuple(repeat(spdBounds, 3 * numwaves)) + tuple(repeat((MIN_REFLECTANCE, 2.0), 3 * numwaves)))
    # format: 3 spectral primaries spd, xyz transform matrix
    initialGuess = np.concatenate((np.repeat((MAX_REFLECTANCE),
        (numwaves * 3)), np.repeat(1.0, (numwaves * 3))))
    print("initial guess is", initialGuess)

    result = differential_evolution(
        objectiveFunction,
        x0=initialGuess,
        callback=plotProgress,
        bounds=bounds,
        workers=workers,
        mutation=(0.1, 1.99),
        maxiter=maxiter,
        tol=tol,
        popsize=npop,
        polish=True,
        disp=True
    ).x

    (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz,
            sds, illuminant_xyz, tmat) = processResult(result)

    mspds = []
    if solveAdditionalXYZs:
        mspds = []
        for targetXYZ in additionalXYZs:
            boundsSingle = tuple(repeat(spdBounds, numwaves))
            initialGuess = np.repeat(MAX_REFLECTANCE, numwaves)
            result = differential_evolution(
                objectiveFunctionSingle,
                x0=initialGuess,
                bounds=boundsSingle,
                args=(targetXYZ, spectral_to_XYZ_m),
                workers=workers,
                mutation=(0.1, 1.99),
                maxiter=maxiter,
                popsize=npop,
                disp=True).x
            mspd = SpectralDistribution((result), waves)
            mspd.name = str(targetXYZ)
            mspds.append(result)

    print("optimal (maybe) wavelengths:", np.array2string(waves, separator=', '))

    print("rgb_to_foo_m is")
    print(np.array2string(sds, separator=', '))

    print("foo_to_XYZ_m is")
    print(np.array2string(spectral_to_XYZ_m, separator=', '))

    print("foo_to_RGB_m is")
    print(np.array2string(spectral_to_RGB_m, separator=', '))

    print("foo_to_Device_RGB_m is")
    print(np.array2string(Spectral_to_Device_RGB_m, separator=', '))

    if solveAdditionalXYZs:
        print("munsell/additional colors are")
        for mspd in mspds:
            print(np.array2string(mspd, separator=', '))

    
    if plotMixes:
        draw_primaries(spectral_to_XYZ_m)
        plotColorMixes(spectral_to_XYZ_m, Spectral_to_Device_RGB_m, sds)
