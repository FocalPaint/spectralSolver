import colour
import numpy as np


# this is our model colorspace that was want to be multispectral
# it's much easier to use a smaller gamut like sRGB, than wider gamutes
colorspace = colour.models.RGB_COLOURSPACE_P3_D65
#colorspace = colour.models.RGB_COLOURSPACE_sRGB

# This is our target RGB colorspace
# probably the same as our model
#colorspacetarget = colour.models.RGB_COLOURSPACE_sRGB
colorspacetarget = colour.models.RGB_COLOURSPACE_P3_D65

# iOS/UIkit Metal uses sRGB primaries even if it is wide color <extended sRGB?>
# use this for transforming to your device's screen if it needs to be different
# from the model colorspace
colorspaceTargetDevice = colour.models.RGB_COLOURSPACE_sRGB


# wavelength range to solve for
# maybe makes sense to make this as narrow as possible
# maybe not
begin = 380.0 
end = 710.0

# number of wavelengths/channels to solve for
# wavelengths will be non-uniformly spaced
# and will penalize solutions with less than waveVariance
# distance between channels
numwaves = 12
waveVariance = 2.0

# max iterations for solver
maxiter = 2000
# population size for diffev
npop = 20

# solve additional colors? see munsell.py
# and add your own XYZ colors
solveAdditionalXYZs = False

additionalXYZs = ([ 0.46780336,  0.23689442,  0.07897962], [ 0.60375823,  0.48586636,  0.08183366], [ 0.69141481,  0.72890368,  0.03672838], [ 0.53874774,  0.74048729,  0.04405483], [ 0.36800563,  0.72124238,  0.52510832], [ 0.45262124,  0.75488848,  0.9837921 ], [ 0.38936903,  0.52007146,  1.16368056], [ 0.47838485,  0.48171774,  1.15655669], [ 0.58214621,  0.39101099,  1.1441827 ], [ 0.59798203,  0.31675163,  0.46063757])

# plot color mixes and spectral curves
# (matplotlib)
plotMixes = True


illuminant_xy = colour.CCS_ILLUMINANTS['cie_2_1931']['D65'].copy()
illuminant_XYZ = colour.xy_to_XYZ(illuminant_xy)
illuminant_SPD = colour.SDS_ILLUMINANTS['D65'].copy()

# list of colors to plot mixes with
colorset = np.array([[0.01, 0.01, 0.01], # black
                   [1., 1., 1.],  # white
                   [0., 0., 1.],  # blue
                   [0., 1., 0.],  # green
                   [1., 0., 0.],  # red
                   [0., 1., 1.],  # cyan
                   [1., 1., 0.],  # yellow
                   [1., 0., 1.],  # magenta
                   [1., 0.18, 0.], # orange
                   [1., 0., 0.18], # fuscia
                   [0.18, 1., 0.], # lime 
                   [0.18, 0., 1.], #purple
                   [0., .18, 1.],  #sky blue
                   [0., 1., 0.18]])#sea foam  


# should not have to edit below
colorspace.use_derived_transformation_matrices(True)
colorspacetarget.use_derived_transformation_matrices(True)

RGB_to_XYZ_m = colorspace.matrix_RGB_to_XYZ
XYZ_to_RGB_m = colorspacetarget.matrix_XYZ_to_RGB

XYZ_to_RGB_Device_m = colorspaceTargetDevice.matrix_XYZ_to_RGB

# conversions from light emission to reflectance must avoid absolute zero
# because log(0.0) is undefined
WGM_EPSILON = .0001

CMFS = colour.MSDS_CMFS['cie_2_1931'].copy()
