from settings import *
from tools import *
from colour import (XYZ_to_xy, SpectralDistribution)
from os.path import exists
from os import remove

# layout of the minimization parameter vector:
# <primarySPDs->wavelengths(nm)


red_XYZ =  colour.RGB_to_XYZ([1.0,0.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([0.0,1.0,0.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([0.0,0.0,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)

XYZ = [red_XYZ, green_XYZ, blue_XYZ]

def extractSPDS(a, numwaves):
    sds = np.asarray(a)[:4 * numwaves].reshape((4, numwaves))
    return sds
    
def extractCMFS(a, numwaves):
    cmfs = np.array([[]])
    waves = np.sort(np.asarray(a)[4 * numwaves:5 * numwaves])
    for idx, wave in enumerate(waves):
        if idx == 0:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=1)
        else:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=0)
    return cmfs

def extractDataFromParameter(a):
    sds = extractSPDS(a, numwaves)
    cmfs = extractCMFS(a, numwaves)
    illuminant = sds[3]
    tmat = generateT_MATRIX_XYZ(cmfs, illuminant)

    return (sds, cmfs, tmat)



def processResult(a):
    sds, cmfs, tmat = extractDataFromParameter(a)
    spectral_to_XYZ_m = generateT_MATRIX_XYZ(cmfs, sds[3])
    spectral_to_RGB_m = generateT_MATRIX_RGB(cmfs, sds[3], XYZ_to_RGB_m)
    Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, sds[3], XYZ_to_RGB_Device_m)
    waves = np.sort(np.asarray(a)[4 * numwaves:5 * numwaves])
    red_xyz = spectral_to_XYZ(sds[0], spectral_to_XYZ_m)
    green_xyz = spectral_to_XYZ(sds[1], spectral_to_XYZ_m)
    blue_xyz = spectral_to_XYZ(sds[2], spectral_to_XYZ_m)
    illuminant_xyz = spectral_to_XYZ(sds[3], spectral_to_XYZ_m)
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
        (sds[3]),
        waves)
    illuminant_sd.name = str(illuminant_xyz)
    return (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz,
            illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, cmfs, tmat)


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
    waves = np.sort(np.asarray(a)[4 * numwaves:5 * numwaves])
    _, counts = np.unique(waves, return_counts=True)
    if np.any(counts > 1):
        return np.inf
    else:
        return 0.0

# try to have at least some difference between each wavelength
# penalize less than waveVariance
def varianceWaves(a):
    waves = np.sort(np.asarray(a)[4 * numwaves:5 * numwaves])
    variance = np.min(np.diff(np.sort(waves)))
    if variance < waveVariance:
        return (waveVariance - variance)
    else:
        return 0.0

def plotProgress(xk, convergence):
    (waves, spectral_to_XYZ_m, spectral_to_RGB_m, Spectral_to_Device_RGB_m, red_xyz, green_xyz, blue_xyz, 
    illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, cmfs, tmat) = processResult(xk)
    red_delta = np.linalg.norm(red_xyz - XYZ[0])
    green_delta = np.linalg.norm(green_xyz - XYZ[1])
    blue_delta = np.linalg.norm(blue_xyz - XYZ[2])
    ilum_delta = np.linalg.norm(illuminant_xy - colour.XYZ_to_xy(illuminant_xyz))
    bumpiness = minimize_slopes([red_sd.values, green_sd.values, blue_sd.values])
    variance = varianceWaves(xk)
    illum_bumpiness = minimize_slope(illuminant_sd.values)
    yellow = red_sd.values + green_sd.values
    mixtest1 = mix_test(blue_sd.values, yellow, green_sd.values, 0.5, spectral_to_XYZ_m)
    cyan = blue_sd.values + green_sd.values + red_sd.values * 0.05
    mixtest2 = mix_test(blue_sd.values, np.repeat(1.0, numwaves), cyan, 0.5, spectral_to_XYZ_m)
    purple = red_sd.values + blue_sd.values
    mixtest3 = mix_test(red_sd.values, blue_sd.values, purple, 0.5, spectral_to_XYZ_m)
    # darkp = red_sd.values* 0.298 + green_sd.values * 0.18 + blue_sd.values * 0.551
    # lightcy = red_sd.values * 0.502 + green_sd.values * 0.723 + blue_sd.values * 0.861
    # mixtest4 = mix_test(darkp, np.repeat(1.0, numwaves), lightcy, 0.5, spectral_to_XYZ_m) 	
    lum_drop_rg = luminance_drop(red_sd.values, green_sd.values, 0.5, spectral_to_XYZ_m)
    lum_drop_rb= luminance_drop(red_sd.values, blue_sd.values, 0.5, spectral_to_XYZ_m)
    lum_drop_gb = luminance_drop(green_sd.values, blue_sd.values, 0.5, spectral_to_XYZ_m)
    vis_efficiency = np.sum(cmfs, axis=0)[1]
    sums = ((np.sum([red_sd.values, green_sd.values, blue_sd.values],axis=0) - 1.0) ** 2.0).sum()

    print("cost metric (smaller = better), weighted cost, actual cost value")
    print("red delta:       ", red_delta ** 2.0 * weight_red, red_delta)
    print("green delta:     ", green_delta ** 2.0 * weight_green, green_delta)
    print("blue delta:      ", blue_delta ** 2.0 * weight_blue, blue_delta)
    print("illum xy delta:  ", ilum_delta ** 2.0 * weight_illuminant_white, ilum_delta)
    print("bumpiness:       ", bumpiness * weight_minslope, bumpiness)
    print("wave variance    ", variance * weight_variance, variance)
    print("illum bumpiness  ", illum_bumpiness * weight_ill_slope, illum_bumpiness)
    print("lum drop rg      ", lum_drop_rg ** 2.0 * weight_lum_drop_rg, lum_drop_rg)
    print("lum drop rb      ", lum_drop_rb ** 2.0 * weight_lum_drop_rb, lum_drop_rb)
    print("lum drop gb      ", lum_drop_gb ** 2.0 * weight_lum_drop_gb, lum_drop_gb)
    print("visual effic     ", -(vis_efficiency ** 2.0) * weight_visual_efficiency, -vis_efficiency)
    print("sd sums to one   ", sums * weight_sum_to_one, sums)

    
    print("mix green delta: ",  mixtest1 ** 2.0 * weight_mixtest1, mixtest1)
    # nudge b+w towards desaturated cyan
   
    print("mix bl/wh delta: ",  mixtest2 ** 2.0 * weight_mixtest2, mixtest2)
    print("mix prple delta: ",  mixtest3 ** 2.0 * weight_mixtest3, mixtest3)
    # print("mix dprp/w delta ",  mixtest4 ** 2.0 * weight_mixtest4, mixtest4)
    print("selected wavelengths: ", waves)
    print("`touch halt` to exit early with this solution.")
    print("---")

    if exists("halt"):
        print("halting early. . .")
        remove("halt")
        return True
    else:
        return False