from settings import *
from tools import *
from colour import (XYZ_to_xy, SpectralDistribution)
from os.path import exists
from os import remove
from plotting import *

# layout of the minimization parameter vector:
# <primarySPDs->FOO_TO_XYZ_MATRIX


red_XYZ =  colour.RGB_to_XYZ([1.0,0.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([0.0,1.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([0.0,0.0,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
white_XYZ = colour.RGB_to_XYZ([1.0,1.0,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)

XYZ = [red_XYZ, green_XYZ, blue_XYZ]

def extractSPDS(a, numwaves):
    sds = np.asarray(a)[:3 * numwaves].reshape((3, numwaves))
    return sds
    
def extractTMAT(a, numwaves):
    # print(np.asarray(a).shape)
    tmat = np.asarray(a)[3 * numwaves:7 * numwaves].reshape((3, numwaves))
    # print(tmat)
        # waves = np.sort(np.asarray(a)[4 * numwaves:5 * numwaves])

    return tmat

def extractDataFromParameter(a):
    sds = extractSPDS(a, numwaves)
    tmat = extractTMAT(a, numwaves)

    return (sds, tmat)



def processResult(a):
    sds, tmat = extractDataFromParameter(a)
    spectral_to_XYZ_m = tmat
    spectral_to_RGB_m = generateT_MATRIX_RGB(tmat, XYZ_to_RGB_m)
    Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(tmat, XYZ_to_RGB_Device_m)
    waves = np.linspace(1,numwaves,numwaves)
    red_xyz = spectral_to_XYZ(sds[0], spectral_to_XYZ_m)
    green_xyz = spectral_to_XYZ(sds[1], spectral_to_XYZ_m)
    blue_xyz = spectral_to_XYZ(sds[2], spectral_to_XYZ_m)
    illuminant_xyz = spectral_to_XYZ(np.repeat(1.0, numwaves), spectral_to_XYZ_m)
    # illuminant_sd.name = str(illuminant_xyz)
    return (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz,
            sds, illuminant_xyz, tmat)


def mix_test(sda, sdb, targetsd, ratio, tmat):
    mixed = spectral_Mix_WGM(sda, sdb, ratio)
    mixedXYZ = spectral_to_XYZ(mixed, tmat)
    mixedxy = XYZ_to_xy(mixedXYZ)
    targetXYZ = spectral_to_XYZ(targetsd, tmat)
    targetxy = XYZ_to_xy(targetXYZ)

    diff = np.linalg.norm(mixedxy - targetxy)
    return diff

def mix_test_RGB(sda, sdb, targetRGB, ratio, tmat):
    mixed = spectral_Mix_WGM(sda, sdb, ratio)
    mixedXYZ = spectral_to_XYZ(mixed, tmat)
    mixedxy = XYZ_to_xy(mixedXYZ)
    targetXYZ = colour.RGB_to_XYZ(targetRGB, illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
    targetxy = XYZ_to_xy(targetXYZ)

    diff = np.linalg.norm(mixedxy - targetxy)
    return diff

def mix_test_RGB_RGB(rgbA, rgbB, targetRGB, ratio, tmat, sds):
    sda = rgb_to_Spectral(rgbA, sds)
    sdb = rgb_to_Spectral(rgbB, sds)
    mixed = spectral_Mix_WGM(sda, sdb, ratio)
    mixedXYZ = spectral_to_XYZ(mixed, tmat)
    mixedxy = XYZ_to_xy(mixedXYZ)
    targetXYZ = colour.RGB_to_XYZ(targetRGB, illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
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
