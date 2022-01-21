import colour
import numpy as np
import sys

from colour.plotting import *

from colour import (XYZ_to_xy, SpectralDistribution)

from scipy.optimize import differential_evolution, basinhopping
from sympy import true
from solver import *
from plotting import plotSDS, plotColorMixes
from tools import generateT_MATRIX_RGB
from itertools import repeat
from os.path import exists
from os import remove

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)

red_XYZ =  colour.RGB_to_XYZ([1.0,0.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([0.0,1.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([0.0,0.0,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)

XYZ = [red_XYZ, green_XYZ, blue_XYZ]

def func(a):
    return 0.0
sd = np.repeat(0.0, numwaves)

def extractDataFromParameter(a):
    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    # illuminant is plucked from canonical SD, but likely won't match xy
    illuminantOriginal = extractIlluminantSPD(a, numwaves)
    illuminantModifer = extractIlluminantModifier(a, numwaves)
    # jitter the illuminant a bit in hopes of matching the right chromaticity xy
    illuminant = np.multiply(illuminantOriginal, illuminantModifer)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)

    return (sds, cmfs, illuminantOriginal, illuminant, tmat)

def processResult(a):
    sds, cmfs, illuminantOriginal, illuminant, tmat = extractDataFromParameter(a)
    spectral_to_XYZ_m = generateT_MATRIX_XYZ(cmfs, illuminant)
    spectral_to_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)
    Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_Device_m)
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    red_xyz = spectral_to_XYZ(sds[0], spectral_to_XYZ_m)
    green_xyz = spectral_to_XYZ(sds[1], spectral_to_XYZ_m)
    blue_xyz = spectral_to_XYZ(sds[2], spectral_to_XYZ_m)
    illuminant_xyz = spectral_to_XYZ(illuminant, spectral_to_XYZ_m)
    # print("final XYZ results:", red_xyz, green_xyz, blue_xyz, illuminant_xyz)
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
    return (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz, 
            illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, illuminantOriginal)

def objectiveFunction(a):

    sds, cmfs, illuminantOriginal, illuminant, tmat = extractDataFromParameter(a)

    result = minimize_slopes(sds) * weight_minslope
    result += match_XYZ(sds[0], XYZ[0], tmat) ** 2.0 * weight_red
    result += match_XYZ(sds[1], XYZ[1], tmat) ** 2.0 * weight_green
    result += match_XYZ(sds[2], XYZ[2], tmat) ** 2.0 * weight_blue
    result += match_xy(illuminant, illuminant_XYZ, tmat) ** 2.0 * weight_illumiant
    result += varianceWaves(a) * weight_variance
    result += uniqueWaves(a) * weight_uniqueWaves

    # penalize difference from original illuminant sd
    result += np.absolute(illuminant - illuminantOriginal).sum() * weight_illuminant_shape

    # penalize non-smooth illuminant
    result += minimize_slope(illuminant) * weight_ill_slope
    
    # nudge b+y = green
    yellow = sds[0] + sds[1]
    result += mix_test(sds[2], yellow, sds[1], 0.5, tmat) ** 2.0 * weight_mixtest1
    # nudge b+w towards desaturated cyan
    cyan = sds[1] + sds[2] + (sds[0] * 0.05)
    result += mix_test(sds[2], np.repeat(1.0, numwaves), cyan, 0.5, tmat) ** 2.0 * weight_mixtest2
    # nudge b+r should be purple
    purple = sds[0] + sds[2]
    result += mix_test(sds[0], sds[2], purple, 0.5, tmat) ** 2.0 * weight_mixtest3

    # penalize large drop in luminance when mixing primaries
    result += luminance_drop(sds[0], sds[1], 0.5, tmat) ** 2.0 * weight_lum_drop_rg
    result += luminance_drop(sds[0], sds[2], 0.5, tmat) ** 2.0 * weight_lum_drop_rb
    result += luminance_drop(sds[1], sds[2], 0.5, tmat) ** 2.0 * weight_lum_drop_gb
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

def luminance_drop(sda, sdb, ratio, tmat):
    mixed = spectral_Mix_WGM(sda, sdb, ratio)
    mixedXYZ = spectral_to_XYZ(mixed, tmat)
    xyzA = spectral_to_XYZ(sda, tmat)
    xyzB = spectral_to_XYZ(sdb, tmat)
    lumAvg = np.mean((xyzA, xyzB), axis=0)[1]

    diff = lumAvg - mixedXYZ[1]
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
    diff = np.sum(np.diff(np.asarray(a)) ** 2)
    return  diff

def minimize_slopes(sds):
    """
    minimize multiple slopes
    """
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

def plotProgress(xk, convergence):
    (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz, 
    illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, illuminantOriginal) = processResult(xk)
    # plotSDS([red_sd, green_sd, blue_sd], illuminant_sd)
    red_delta = np.linalg.norm(red_xyz - XYZ[0])
    green_delta = np.linalg.norm(green_xyz - XYZ[1])
    blue_delta = np.linalg.norm(blue_xyz - XYZ[2])
    ilum_delta = np.linalg.norm(illuminant_xy - colour.XYZ_to_xy(illuminant_xyz))
    bumpiness = minimize_slopes([red_sd.values, green_sd.values, blue_sd.values])
    variance = varianceWaves(xk)
    illum_shape = np.absolute(illuminant_sd.values - illuminantOriginal).sum()
    illum_bumpiness = minimize_slope(illuminant_sd.values)
    yellow = red_sd.values + green_sd.values
    mixtest1 = mix_test(blue_sd.values, yellow, green_sd.values, 0.5, spectral_to_XYZ_m)
    cyan = blue_sd.values + green_sd.values + red_sd.values * 0.05
    mixtest2 = mix_test(blue_sd.values, np.repeat(1.0, numwaves), cyan, 0.5, spectral_to_XYZ_m)
    lum_drop_rg = luminance_drop(red_sd.values, green_sd.values, 0.5, spectral_to_XYZ_m)
    lum_drop_rb= luminance_drop(red_sd.values, blue_sd.values, 0.5, spectral_to_XYZ_m)
    lum_drop_gb = luminance_drop(green_sd.values, blue_sd.values, 0.5, spectral_to_XYZ_m)

    print("cost metric, weighted delta cost, actual delta value")
    print("red delta:       ", red_delta ** 2.0 * weight_red, red_delta)
    print("green delta:     ", green_delta ** 2.0 * weight_green, green_delta)
    print("blue delta:      ", blue_delta ** 2.0 * weight_blue, blue_delta)
    print("illum xy delta:  ", ilum_delta ** 2.0 * weight_illumiant, ilum_delta)
    print("bumpiness:       ", bumpiness * weight_minslope, bumpiness)
    print("wave variance    ", variance * weight_variance, variance)
    print("illum shape diff ", illum_shape * weight_illuminant_shape, illum_shape )
    print("illum bumpiness  ", illum_bumpiness * weight_ill_slope, illum_bumpiness)
    print("lum drop rg      ",  lum_drop_rg ** 2.0 * weight_lum_drop_rg, lum_drop_rg)
    print("lum drop rb      ",  lum_drop_rb ** 2.0 * weight_lum_drop_rb, lum_drop_rb)
    print("lum drop gb      ",  lum_drop_gb ** 2.0 * weight_lum_drop_gb, lum_drop_gb)

    
    print("mix green delta: ",  mixtest1 ** 2.0 * weight_mixtest1, mixtest1)
    # nudge b+w towards desaturated cyan
   
    print("mix bl/wh delta: ",  mixtest2 ** 2.0 * weight_mixtest2, mixtest2)
    print("`touch halt` to exit early with this solution.")
    print("---")

    if exists("halt"):
        print("halting early. . .")
        remove("halt")
        return True
    else:
        return False

from settings import *
if __name__ == '__main__':
    spdBounds = (WGM_EPSILON, 1.0 - WGM_EPSILON)
    waveBounds = (begin, end)
    illuminantModifierBounds = (0.75, 2.0)
    from itertools import repeat
    bounds = (tuple(repeat(spdBounds, 3 * numwaves)) +
                  tuple(repeat(waveBounds, numwaves)) +
                  tuple(repeat(illuminantModifierBounds, numwaves)))
    # format: 3 spectral primaries + wavelength indices in nm, + illuminant modifiers (%)
    initialGuess = np.concatenate((np.repeat((1.0 - WGM_EPSILON),
        (numwaves * 3)), np.linspace(begin, end, num=numwaves, endpoint=True),
        np.repeat(1.0, numwaves)))
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
        illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, illuminantOriginal) = processResult(result)

    mspds = []
    if solveAdditionalXYZs:
        mspds = []
        for targetXYZ in additionalXYZs:
            boundsSingle = tuple(repeat(spdBounds, numwaves))
            initialGuess = np.repeat(1.0 - WGM_EPSILON, numwaves)
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