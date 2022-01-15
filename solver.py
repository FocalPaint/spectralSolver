from colour.models.rgb.transfer_functions import linear
from mystic.constraints import unique
import numpy as np

from colour.colorimetry import (SpectralDistribution)

from colour import XYZ_to_xy
from mystic.solvers import diffev
from mystic.monitors import VerboseMonitor

from scipy.optimize import differential_evolution, LinearConstraint, Bounds

from settings import *
from tools import *
from itertools import repeat


# layout of the minimization parameter vector:
# <primarySPDs->wavelengths(nm)

def extractSPDS(a, numwaves):
    sds = np.asarray(a)[:3 * numwaves].reshape((3, numwaves))
    return sds
    
def extractCMFS(a, numwaves):
    cmfs = np.array([[]])
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    for idx, wave in enumerate(waves):
        if idx == 0:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=1)
        else:
            cmfs = np.append(cmfs, [CMFS[wave]], axis=0)
    return cmfs

def extractIlluminantSPD(a, numwaves):

    illuminantSPD = np.array([])
    waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
    for wave in waves:
        illuminantSPD = np.append(illuminantSPD, illuminant_SPD[wave])
    return illuminantSPD


# def XYZ_to_spectral_colorspace(
#         XYZ):

#     XYZ = (XYZ)
#     print("going to try to match ", XYZ)

#     sd = np.repeat(0.0, numwaves)

#     global objectiveFunction
#     def objectiveFunction(a):
#         result = minimize_slope(a)
#         result += match_red(a)
#         result += match_green(a)
#         result += match_blue(a)
#         result += match_white(a)
#         result += varianceWaves(a)
#         result += uniqueWaves(a)
#         return result

#     def minimize_slope(a):
#         """
#         Objective function.
#         """
#         sds = np.exp(extractSPDS(a, numwaves))
#         red_diff = np.sum(np.diff(sds[0]) ** 2)
#         green_diff = np.sum(np.diff(sds[1]) ** 2)
#         blue_diff = np.sum(np.diff(sds[2]) ** 2)
#         diff = red_diff + green_diff + blue_diff
#         return  diff

#     def match_red(a):
#         """
#         Function defining the constraint.
#         """
#         sds = extractSPDS(a, numwaves)
#         cmfs = extractCMFS(a, numwaves)
#         illuminant = extractIlluminantSPD(a, numwaves)
#         tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
#         sd[:] = np.exp(sds[0])
#         xyz = Spectral_to_XYZ(sd, tmat)
        
#         diff = np.linalg.norm(xyz - XYZ[0])
#         return diff * 1000.

#     def match_green(a):
#         """
#         Function defining the constraint.
#         """
#         sds = extractSPDS(a, numwaves)
#         cmfs = extractCMFS(a, numwaves)
#         illuminant = extractIlluminantSPD(a, numwaves)
#         tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
#         sd[:] = np.exp(sds[1])
#         xyz = Spectral_to_XYZ(sd, tmat)

#         diff = np.linalg.norm(xyz - XYZ[1])
#         return diff * 1000.

#     def match_blue(a):
#         """
#         Function defining the constraint.
#         """
#         sds = extractSPDS(a, numwaves)
#         cmfs = extractCMFS(a, numwaves)
#         illuminant = extractIlluminantSPD(a, numwaves)
#         tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
#         sd[:] = np.exp(sds[2])
#         xyz = Spectral_to_XYZ(sd, tmat)
        
#         diff = np.linalg.norm(xyz - XYZ[2])
#         return diff * 1000.

#     # we want the selected wavelengths extracted
#     # from the illuminant SPD to still match the xy
#     # so that a perfect reflector is still white
#     def match_white(a):
#         """
#         Function defining the constraint.
#         """
#         cmfs = extractCMFS(a, numwaves)
#         illuminant = extractIlluminantSPD(a, numwaves)
#         tmat = generateT_MATRIX_XYZ(cmfs, illuminant)
#         xyz = Spectral_to_XYZ(illuminant, tmat)
#         xy = XYZ_to_xy(xyz)
        
#         diff = np.linalg.norm(xy - illuminant_xy)
#         return diff * 1000.

#     # this is kind of impossible without a lot of wavelengths
#     # def sum_to_one(a):
#     #     """
#     #     constrain to conserve energy. sum of r+g+b must == 1.0
#     #     """
#     #     sds = extractSPDS(a, numwaves)
#     #     sums = ((sds - 1.0)**2.0).sum()
#     #     return sums
    
#     # shouldn't need mix tests if things work out. . .
#     # def mix_test(a):
#     #     # try to get a mix with yellow+blue == green
#     #     sds = extractSPDS(a, numwaves)
#     #     cmfs = extractCMFS(a, numwaves)
#     #     illuminant = extractIlluminantSPD(a, numwaves)
#     #     T_MATRIX = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)
#     #     #illuminant = SpectralDistribution(
#     #     #(np.exp(sds[3])),
#     #     #wavelengths)
#     #     blue = sds[2]
#     #     yellow = np.log(np.exp(sds[0]) + np.exp(sds[1]))
#     #     green = Spectral_Mix_WGM(yellow, blue, 0.5)
        
#     #     green_rgb = Spectral_to_RGB(green, T_MATRIX)
#     #     diff = np.linalg.norm(green_rgb - np.array([0.2,1.,0.2]))
#     #     return diff

#     # def mix_test2(a):
#     #     # try to nudge blue+white toward cyan instead of violet
#     #     sds = extractSPDS(a, numwaves)
#     #     cmfs = extractCMFS(a, numwaves)
#     #     illuminant = extractIlluminantSPD(a, numwaves)
#     #     T_MATRIX = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)
#     #     #illuminant = SpectralDistribution(
#     #     #(np.exp(sds[3])),
#     #     #wavelengths)
#     #     blue = sds[2]
#     #     white = np.log(np.exp(sds[0]) + np.exp(sds[1]) + np.exp(sds[2]))
#     #     cyan = Spectral_Mix_WGM(blue, white, 0.5)
#     #     cyan_rgb = Spectral_to_RGB(cyan, T_MATRIX)
#     #     diff = np.linalg.norm(cyan_rgb - np.array([0.5,1.,1.]))
#     #     return diff


#     # having duplicate wavelengths is an error for Colour library
#     # (and probably doesn't make sense)
#     def uniqueWaves(a):
#         waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
#         _, counts = np.unique(waves, return_counts=True)
#         if np.any(counts > 1):
#             return np.inf
#         else:
#             return 0.0

#     # try to have at least some difference between each wavelength
#     # penalize less than waveVariance
#     def varianceWaves(a):
#         waves = np.sort(np.asarray(a)[3 * numwaves:4 * numwaves])
#         variance = np.min(np.diff(np.sort(waves)))
#         if variance < waveVariance:
#             return (waveVariance - variance) * 100.
#         else:
#             return 0.0

#     from mystic.penalty import  linear_equality
#     from mystic.constraints import impose_bounds
#     from mystic.tools import chain

    
#     matchr = linear_equality(match_red)
#     matchg = linear_equality(match_green)
#     matchb = linear_equality(match_blue)
#     uniqWaves = linear_equality(uniqueWaves)
#     varyWaves = linear_equality(varianceWaves)
#     matchw = linear_equality(match_white)
#     # sumone = linear_equality(sum_to_one)
#     # mixtest = linear_equality(mix_test)
#     # mixtest2 = linear_equality(mix_test2)

#     @chain(matchr)#, matchg, matchb, matchw, uniqWaves, varyWaves)
#     def penalty(x):
#         return 0.0

#     #log values, constrain solution to >0 and less than exp(0.5).
#     #should really be -10 thru 0 but hard to solve
#     spdBounds = (-12, -0.00001)
#     waveBounds = (begin, end)
#     bounds = tuple(repeat(spdBounds, 3 * numwaves)) + tuple(repeat(waveBounds, numwaves))
#     print(bounds)
#     @impose_bounds(bounds)
#     def simple(a):
#         return a

#     stepmon = VerboseMonitor(1, xinterval=10)

#     initialGuess = np.concatenate((np.random.rand(numwaves * 3) * -10 - 0.00001, np.random.rand(numwaves) * (end - begin) + begin))
#     print(initialGuess)

#     constraints = ({'type': 'eq', 'fun': match_red},
#                    {'type': 'eq', 'fun': match_green},
#                    {'type': 'eq', 'fun': match_blue},
#                    {'type': 'eq', 'fun': match_white},
#                    {'type': 'eq', 'fun': uniqueWaves},
#                    {'type': 'eq', 'fun': varianceWaves})

#     result = differential_evolution(
#         objectiveFunction,
#         bounds=bounds,
#         x0=initialGuess,
#         workers=1,
#         disp=True).x

#     # result = diffev(
#     #     minimize_slope,
#     #     initialGuess,
#     #     bounds=bounds,
#     #     penalty=penalty,
#     #     npop=npop,
#     #     itermon=stepmon,
#     #     constraints=simple,
#     #     ftol=1e-8,
#     #     gtol=100,
#     #     maxiter=maxiter
#     #     )
#     sds = np.exp(extractSPDS(result, numwaves))

#     waves = np.sort(np.asarray(result)[3 * numwaves:4 * numwaves])
#     cmfs = extractCMFS(result, numwaves)
#     illuminant = extractIlluminantSPD(result, numwaves)
#     T_MATRIX_XYZ = generateT_MATRIX_XYZ(cmfs, illuminant)
#     T_MATRIX_RGB = generateT_MATRIX_RGB(cmfs, illuminant, XYZ_to_RGB_m)

#     print("original XYZ targets: ", XYZ)
#     red_xyz = Spectral_to_XYZ(sds[0], T_MATRIX_XYZ)
#     green_xyz = Spectral_to_XYZ(sds[1], T_MATRIX_XYZ)
#     blue_xyz = Spectral_to_XYZ(sds[2], T_MATRIX_XYZ)
#     illuminant_xyz = Spectral_to_XYZ(illuminant, T_MATRIX_XYZ)
#     print("final XYZ results:", red_xyz, green_xyz, blue_xyz, illuminant_xyz)
#     red_sd = SpectralDistribution(
#         (sds[0]),
#         waves)
#     red_sd.name = str(red_xyz)
#     green_sd = SpectralDistribution(
#         (sds[1]),
#         waves)
#     green_sd.name = str(green_xyz)
#     blue_sd = SpectralDistribution(
#         (sds[2]),
#         waves)
#     blue_sd.name = str(blue_xyz)
#     illuminant_sd = SpectralDistribution(
#         (illuminant),
#         waves)
#     illuminant_sd.name = str(illuminant_xyz)
#     return (red_sd, green_sd, blue_sd, waves, illuminant_sd, cmfs, T_MATRIX_XYZ, T_MATRIX_RGB)



# def XYZ_to_spectral_1(
#         XYZ,
#         T_MATRIX,
#         waves):

#     print("going to try to match ", XYZ)

#     sd = np.repeat(0.0, numwaves)

#     def minimize_slope(a):
#         """
#         Objective function.
#         """
#         diff = np.sum(np.diff(np.exp(np.asarray(a)) ** 2))
#         return  diff

#     def match_XYZ(a):
#         """
#         Function defining the constraint.
#         """
#         sd[:] = np.exp(np.asarray(a))
#         xyz = Spectral_to_XYZ(sd, T_MATRIX)
#         diff = np.linalg.norm(xyz - XYZ)
#         return diff * 10.

#     from mystic.penalty import  linear_equality, linear_equality, barrier_inequality, quadratic_equality, lagrange_equality, lagrange_inequality
#     from mystic.constraints import impose_bounds, as_constraint
#     from mystic.tools import chain    
#     matchXYZ = linear_equality(match_XYZ)

#     @chain(matchXYZ)
#     def penalty(x):
#         return 0.0

#     #log values, constrain solution to >0 and less than exp(0.5).
#     #should really be -10 thru 0 but hard to solve
#     spdBounds = (-12, -0.00001)
#     waveBounds = (begin, end)
#     bounds = tuple(repeat(spdBounds, numwaves))
#     print(bounds)
#     @impose_bounds(bounds)
#     def simple(a):
#         return a

#     stepmon = VerboseMonitor(1, xinterval=10)

#     initialGuess = np.repeat(0.0, numwaves )
#     print(initialGuess)

#     result = diffev(
#         minimize_slope,
#         initialGuess,
#         bounds=bounds,
#         penalty=penalty,
#         npop=npop,
#         itermon=stepmon,
#         constraints=simple,
#         ftol=1e-8,
#         gtol=2000,
#         maxiter=maxiter
#         )
#     sdResult = np.exp(result)

#     print("original XYZ target: ", XYZ)
#     result_xyz = Spectral_to_XYZ(sdResult, T_MATRIX)
   
#     print("final XYZ results:", result_xyz)
#     #print(np.exp(sds[0]), waves)
#     result_sd = SpectralDistribution(
#         (sdResult),
#         waves)
#     result_sd.name = str(result_xyz)
#     return (result_sd)