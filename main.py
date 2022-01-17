import colour
import numpy as np
import sys

from colour.plotting import *

from colour import (XYZ_to_xy, SpectralDistribution)

from scipy.optimize import differential_evolution, basinhopping
from solver import *
from plotting import plotSDS, plotColorMixes
from tools import generateT_MATRIX_RGB
from itertools import repeat

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)

red_XYZ =  colour.RGB_to_XYZ([1.0,0.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([0.0,1.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([0.0,0.0,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)

XYZ = [red_XYZ, green_XYZ, blue_XYZ]

def func(a):
    return 0.0
sd = np.repeat(0.0, numwaves)

def objectiveFunction(a):

    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    illuminant = extractIlluminantSPD(a, numwaves)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)

    result = minimize_slopes(a) / 100.
    result += match_XYZ(sds[0], XYZ[0], tmat) * 10000.
    result += match_XYZ(sds[1], XYZ[1], tmat) * 10000.
    result += match_XYZ(sds[2], XYZ[2], tmat) * 100000.
    result += match_xy(illuminant, illuminant_XYZ, tmat) * 10000.
    result += varianceWaves(a) / 100.
    result += uniqueWaves(a)
    
    # nudge b+y = green
    yellow = sds[0] + sds[1]
    result += mix_test(sds[2], yellow, sds[1], 0.5, tmat) * 100.
    # nudge b+w towards cyan
    cyan = sds[1] + sds[2]
    result += mix_test(sds[2], np.repeat(1.0, numwaves), cyan, 0.5, tmat) * 100.
    return result


def objectiveFunctionSingle(a, targetXYZ, spectral_to_XYZ_m):
    result = minimize_slope(a)
    result += match_XYZ(a, targetXYZ, spectral_to_XYZ_m) * 10000.
    return result

def mix_test(sda, sdb, targetsd, ratio, tmat):
    mixed = spectral_Mix_WGM(sda, sdb, ratio)
    mixedXYZ = spectral_to_XYZ(mixed, tmat)
    mixedxy = XYZ_to_xy(mixedXYZ)
    targetXYZ = spectral_to_XYZ(targetsd, tmat)
    targetxy = XYZ_to_xy(targetXYZ)

    diff = np.linalg.norm(mixedxy - targetxy)
    return diff

def match_XYZ(a, targetXYZ, spectral_to_XYZ_m):
    """
    match one XYZ
    """
    spec = np.asarray(a)
    xyz = spectral_to_XYZ(spec, spectral_to_XYZ_m)
    diff = np.linalg.norm(xyz - targetXYZ)
    return diff

def match_xy(a, targetXYZ, spectral_to_XYZ_m):
    """
    match one xy
    """
    spec = np.asarray(a)
    xyz = spectral_to_XYZ(spec, spectral_to_XYZ_m)
    xy = XYZ_to_xy(xyz)

    targetxy = XYZ_to_xy(targetXYZ)
    
    diff = np.linalg.norm(xy - targetxy)
    return diff

def minimize_slope(a):
    """
    minimize a slope
    """
    diff = np.sum(np.diff(np.asarray(a) ** 2))
    return  diff

def minimize_slopes(a):
    """
    minimize multipel slopes
    """
    sds = extractSPDS(a, numwaves)
    red_diff = np.sum(np.diff(sds[0]) ** 2)
    green_diff = np.sum(np.diff(sds[1]) ** 2)
    blue_diff = np.sum(np.diff(sds[2]) ** 2)
    diff = red_diff + green_diff + blue_diff
    return  diff

# having duplicate wavelengths is an error for Colour library
# (and probably doesn't make sense)
def uniqueWaves(a):
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    _, counts = np.unique(waves, return_counts=True)
    if np.any(counts > 1):
        return np.inf
    else:
        return 0.0

# try to have at least some difference between each wavelength
# penalize less than waveVariance
def varianceWaves(a):
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    variance = np.min(np.diff(np.sort(waves)))
    if variance < waveVariance:
        return (waveVariance - variance)
    else:
        return 0.0    

from settings import *
if __name__ == '__main__':
    spdBounds = (-12, -0.00001)
    waveBounds = (begin, end)
    from itertools import repeat
    bounds = tuple(repeat(spdBounds, 3 * numwaves)) + tuple(repeat(waveBounds, numwaves))
    initialGuess = np.concatenate((np.repeat(-0.00001, (numwaves * 3)), np.linspace(begin, end, num=numwaves, endpoint=True)))
    print("initial guess is", initialGuess)

    result = differential_evolution(
        objectiveFunction,
        x0=initialGuess,
        bounds=bounds,
        workers=workers,
        mutation=(0.1, 1.99),
        maxiter=maxiter,
        tol=tol,
        popsize=npop,
        polish=True,
        disp=True
    ).x

    sds = extractSPDS(result, numwaves)

    waves = np.sort(np.asarray(result)[3 * numwaves:4 * numwaves])
    cmfs = extractCMFS(result, numwaves)
    illuminant = extractIlluminantSPD(result, numwaves)
    spectral_to_XYZ_m = generateT_MATRIX_XYZ(cmfs, illuminant)
    spectral_to_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)
    Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_Device_m)
    
    print("original XYZ targets: ", XYZ)
    red_xyz = spectral_to_XYZ(sds[0], spectral_to_XYZ_m)
    green_xyz = spectral_to_XYZ(sds[1], spectral_to_XYZ_m)
    blue_xyz = spectral_to_XYZ(sds[2], spectral_to_XYZ_m)
    illuminant_xyz = spectral_to_XYZ(illuminant, spectral_to_XYZ_m)
    print("final XYZ results:", red_xyz, green_xyz, blue_xyz, illuminant_xyz)
    red_sd = SpectralDistribution(
        (sds[0]),
        waves)
    red_sd.name = str(red_xyz)
    green_sd = SpectralDistribution(
        (sds[1]),
        waves)
    green_sd.name = str(green_xyz)
    blue_sd = SpectralDistribution(
        (sds[2]),
        waves)
    blue_sd.name = str(blue_xyz)
    illuminant_sd = SpectralDistribution(
        (illuminant),
        waves)
    illuminant_sd.name = str(illuminant_xyz)

    mspds = []
    if solveAdditionalXYZs:
        mspds = []
        for targetXYZ in additionalXYZs:
            boundsSingle = tuple(repeat(spdBounds, numwaves))
            result = differential_evolution(
                objectiveFunctionSingle,
                bounds=boundsSingle,
                args=(targetXYZ, spectral_to_XYZ_m),
                workers=workers,
                mutation=(0.1, 1.99),
                maxiter=maxiter,
                popsize=npop,
                disp=True).x
            #mspd = XYZ_to_spectral_1(np.array(munseltarglTarget), T_MATRIX, waves=waves)
            mspds.append(np.exp(result))

    print("optimal (maybe) wavelengths:", np.array2string(waves, separator=', '))

    print("Spectral red is")
    print(np.array2string(red_sd.values, separator=', '))

    print("Spectral green is")
    print(np.array2string(green_sd.values, separator=', '))

    print("Spectral blue is")
    print(np.array2string(blue_sd.values, separator=', '))

    print("spectral_to_XYZ_m is")
    print(np.array2string(spectral_to_XYZ_m, separator=', '))

    print("spectral_to_RGB_m is")
    print(np.array2string(spectral_to_RGB_m, separator=', '))

    print("Spectral_to_Device_RGB_m is")
    print(np.array2string(Spectral_to_Device_RGB_m, separator=', '))

    if solveAdditionalXYZs:
        print("munsell/additional colors are")
        for mspd in mspds:
            print(np.array2string(mspd, separator=', '))

    
    if plotMixes:
        plotSDS([red_sd, green_sd, blue_sd], illuminant_sd)
        plotColorMixes(spectral_to_XYZ_m, Spectral_to_Device_RGB_m, [red_sd, green_sd, blue_sd])