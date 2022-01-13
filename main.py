import colour
import numpy as np
import sys

from colour.plotting import *

from colour import (colorimetry)

from solver import XYZ_to_spectral_colorspace
from munsell import calcMunsellColors
from plotting import plotSDS, plotColorMixes
from tools import generateT_MATRIX_RGB

np.set_printoptions(formatter={"float": "{:0.15f}".format}, threshold=sys.maxsize)

from settings import *


red_XYZ =  colour.RGB_to_XYZ([1.0,WGM_EPSILON,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
green_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,1.0,WGM_EPSILON], illuminant_xy, illuminant_xy, RGB_to_XYZ_m)
blue_XYZ = colour.RGB_to_XYZ([WGM_EPSILON,WGM_EPSILON,1.0], illuminant_xy, illuminant_xy, RGB_to_XYZ_m) #colour.sRGB_to_XYZ([0,0,1])

XYZs = [red_XYZ, green_XYZ, blue_XYZ]

# actually solve the SPDs
red_sd, green_sd, blue_sd, waves, illuminant, cmfs, Spectral_to_XYZ_m, Spectral_to_RGB_m = XYZ_to_spectral_colorspace(XYZs)
waves = red_sd.wavelengths
Spectral_to_Device_RGB_m = generateT_MATRIX_RGB(cmfs, illuminant.values, XYZ_to_RGB_Device_m)

mspds = []
if solveAdditionalXYZs:
    mspds = calcMunsellColors(Spectral_to_XYZ_m, waves)


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
        print(np.array2string(mspd.values, separator=', '))

if plotMixes:
    plotSDS([red_sd, green_sd, blue_sd], illuminant, mspds)
    plotColorMixes(Spectral_to_XYZ_m, Spectral_to_Device_RGB_m, [red_sd, green_sd, blue_sd])