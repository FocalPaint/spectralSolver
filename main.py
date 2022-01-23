import colour
import numpy as np
import sys

from colour.plotting import *

from scipy.optimize import differential_evolution, basinhopping
from sympy import true
from solver import *
from plotting import plotSDS, plotColorMixes
from tools import generateT_MATRIX_RGB
from itertools import repeat
from settings import *

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)


def func(a):
    return 0.0
sd = np.repeat(0.0, numwaves)



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
    # dark purple and white should go cyanish
    darkp = sds[0] * 0.298 + sds[1] * 0.18 + sds[2] * 0.551
    lightcy = sds[0] * 0.502 + sds[1] * 0.723 + sds[2] * 0.861
    result += mix_test(darkp, np.repeat(1.0, numwaves), lightcy, 0.5, tmat) ** 2.0 * weight_mixtest4	

    # penalize large drop in luminance when mixing primaries
    result += luminance_drop(sds[0], sds[1], 0.5, tmat) ** 2.0 * weight_lum_drop_rg
    result += luminance_drop(sds[0], sds[2], 0.5, tmat) ** 2.0 * weight_lum_drop_rb
    result += luminance_drop(sds[1], sds[2], 0.5, tmat) ** 2.0 * weight_lum_drop_gb

    # encourage maximal visual efficiency ( high Y )
    result += -np.sum(cmfs, axis=0)[1] ** 2.0 * weight_visual_efficiency
    return result


def objectiveFunctionSingle(a, targetXYZ, spectral_to_XYZ_m):
    result = minimize_slope(a)
    result += match_XYZ(a, targetXYZ, spectral_to_XYZ_m) * 10000.
    return result

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
        illuminant_xyz, red_sd, green_sd, blue_sd, illuminant_sd, illuminantOriginal, cmfs) = processResult(result)

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