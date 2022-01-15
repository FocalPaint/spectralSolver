import colour
import numpy as np
import sys

from colour.plotting import *

from colour import (XYZ_to_xy, SpectralDistribution)

from scipy.optimize import differential_evolution, LinearConstraint, Bounds
from solver import *
from plotting import plotSDS, plotColorMixes
from tools import generateT_MATRIX_RGB
from itertools import repeat

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)

red_XYZ =  colour.RGB_to_XYZ([1.0,WGM_EPSILON,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,1.0,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,WGM_EPSILON,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m) #colour.sRGB_to_XYZ([0,0,1])

XYZ = [red_XYZ, green_XYZ, blue_XYZ]

# singleXYZ = None

# Spectral_to_XYZ_m = None

def func(a):
    return 0.0
sd = np.repeat(0.0, numwaves)

def objectiveFunction(a):
    result = minimize_slopes(a)
    result += match_red(a)
    result += match_green(a)
    result += match_blue(a)
    result += match_white(a)
    result += varianceWaves(a)
    result += uniqueWaves(a)
    return result


def objectiveFunctionSingle(a, targetXYZ, Spectral_to_XYZ_m):
    result = minimize_slope(a)
    result += match_XYZ(a, targetXYZ, Spectral_to_XYZ_m)
    return result

def match_XYZ(a, targetXYZ, Spectral_to_XYZ_m):
    """
    match one XYZ
    """
    sd[:] = np.exp(np.asarray(a))
    xyz = Spectral_to_XYZ(sd, Spectral_to_XYZ_m)
    diff = np.linalg.norm(xyz - targetXYZ)
    return diff * 100.

def minimize_slope(a):
    """
    minimize a slope
    """
    diff = np.sum(np.diff(np.exp(np.asarray(a)) ** 2))
    return  diff

def minimize_slopes(a):
    """
    minimize multipel slopes
    """
    sds = np.exp(extractSPDS(a, numwaves))
    red_diff = np.sum(np.diff(sds[0]) ** 2)
    green_diff = np.sum(np.diff(sds[1]) ** 2)
    blue_diff = np.sum(np.diff(sds[2]) ** 2)
    diff = red_diff + green_diff + blue_diff
    return  diff

def match_red(a):
    """
    Function defining the constraint.
    """
    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    illuminant = extractIlluminantSPD(a, numwaves)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
    sd[:] = np.exp(sds[0])
    xyz = Spectral_to_XYZ(sd, tmat)
    
    diff = np.linalg.norm(xyz - XYZ[0])
    return diff * 10000.

def match_green(a):
    """
    Function defining the constraint.
    """
    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    illuminant = extractIlluminantSPD(a, numwaves)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
    sd[:] = np.exp(sds[1])
    xyz = Spectral_to_XYZ(sd, tmat)

    diff = np.linalg.norm(xyz - XYZ[1])
    return diff * 10000.

def match_blue(a):
    """
    Function defining the constraint.
    """
    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    illuminant = extractIlluminantSPD(a, numwaves)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
    sd[:] = np.exp(sds[2])
    xyz = Spectral_to_XYZ(sd, tmat)
    
    diff = np.linalg.norm(xyz - XYZ[2])
    return diff * 10000.

# we want the selected wavelengths extracted
# from the illuminant SPD to still match the xy
# so that a perfect reflector is still white
def match_white(a):
    """
    Function defining the constraint.
    """
    cmfs = extractCMFS(a, numwaves)
    illuminant = extractIlluminantSPD(a, numwaves)
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
    xyz = Spectral_to_XYZ(illuminant, tmat)
    xy = XYZ_to_xy(xyz)
    
    diff = np.linalg.norm(xy - illuminant_xy)
    return diff * 1000.

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
        return (waveVariance - variance) * 100.
    else:
        return 0.0    

from settings import *
if __name__ == '__main__':
    spdBounds = (-12, -0.00001)
    waveBounds = (begin, end)
    bounds = tuple(repeat(spdBounds, 3 * numwaves)) + tuple(repeat(waveBounds, numwaves))
    result = differential_evolution(
            objectiveFunction,
            bounds=bounds,
            workers=-1,
            maxiter=maxiter,
            popsize=npop,
            polish=True,
            disp=True).x

    sds = np.exp(extractSPDS(result, numwaves))

    waves = np.sort(np.asarray(result)[3 * numwaves:4 * numwaves])
    cmfs = extractCMFS(result, numwaves)
    illuminant = extractIlluminantSPD(result, numwaves)
    Spectral_to_XYZ_m = generateT_MATRIX_XYZ(cmfs, illuminant)
    Spectral_to_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)
    Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_Device_m)
    
    print("original XYZ targets: ", XYZ)
    red_xyz = Spectral_to_XYZ(sds[0], Spectral_to_XYZ_m)
    green_xyz = Spectral_to_XYZ(sds[1], Spectral_to_XYZ_m)
    blue_xyz = Spectral_to_XYZ(sds[2], Spectral_to_XYZ_m)
    illuminant_xyz = Spectral_to_XYZ(illuminant, Spectral_to_XYZ_m)
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
                args=(targetXYZ, Spectral_to_XYZ_m),
                workers=-1,
                maxiter=maxiter,
                popsize=npop,
                disp=True).x
            #mspd = XYZ_to_spectral_1(np.array(munseltarglTarget), T_MATRIX, waves=waves)
            mspds.append(result)

    print("optimal (maybe) wavelengths:", np.array2string(waves, separator=', '))

    print("Spectral red is")
    print(np.array2string(red_sd.values, separator=', '))

    print("Spectral green is")
    print(np.array2string(green_sd.values, separator=', '))

    print("Spectral blue is")
    print(np.array2string(blue_sd.values, separator=', '))

    print("Spectral_to_XYZ_m is")
    print(np.array2string(Spectral_to_XYZ_m, separator=', '))

    print("Spectral_to_RGB_m is")
    print(np.array2string(Spectral_to_RGB_m, separator=', '))

    print("Spectral_to_Device_RGB_m is")
    print(np.array2string(Spectral_to_Device_RGB_m, separator=', '))

    if solveAdditionalXYZs:
        print("munsell/additional colors are")
        for mspd in mspds:
            print(np.array2string(mspd, separator=', '))

    
    if plotMixes:
        plotSDS([red_sd, green_sd, blue_sd], illuminant_sd)
        plotColorMixes(Spectral_to_XYZ_m, Spectral_to_Device_RGB_m, [red_sd, green_sd, blue_sd])


# red_XYZ =  colour.RGB_to_XYZ([1.0,WGM_EPSILON,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
# green_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,1.0,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
# blue_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,WGM_EPSILON,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m) #colour.sRGB_to_XYZ([0,0,1])

# XYZs = [red_XYZ, green_XYZ, blue_XYZ]

# # actually solve the SPDs
# if __name__ == '__main__':
#     red_sd, green_sd, blue_sd, waves, illuminant, cmfs, Spectral_to_XYZ_m, Spectral_to_RGB_m = XYZ_to_spectral_colorspace(XYZs)
#     waves = red_sd.wavelengths
#     Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant.values, XYZ_to_RGB_Device_m)

#     mspds = []
#     if solveAdditionalXYZs:
#         mspds = calcMunsellColors(Spectral_to_XYZ_m, waves)




#     if solveAdditionalXYZs:
#         print("munsell/additional colors are")
#         for mspd in mspds:
#             print(np.array2string(mspd.values, separator=', '))

#     if plotMixes:
#         plotSDS([red_sd, green_sd, blue_sd], illuminant, mspds)
#         plotColorMixes(Spectral_to_XYZ_m, Spectral_to_Device_RGB_m, [red_sd, green_sd, blue_sd])