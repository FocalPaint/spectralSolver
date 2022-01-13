import colour
import numpy as np


# this is our model colorspace that was want to be multispectral
# it's much easier to use a smaller gamut like sRGB, than wider gamutes
# colorspace = colour.models.RGB_COLOURSPACE_P3_D65
colorspace = colour.models.RGB_COLOURSPACE_sRGB

# This is our target RGB colorspace
# probably the same as our model
colorspacetarget = colour.models.RGB_COLOURSPACE_sRGB
# colorspacetarget = colour.models.RGB_COLOURSPACE_P3_D65

# iOS/UIkit Metal uses sRGB primaries even if it is wide color <extended sRGB?>
# use this for transforming to your device's screen if it needs to be different
# from the model colorspace
colorspaceTargetDevice = colour.models.RGB_COLOURSPACE_sRGB


# wavelength range to solve for
# maybe makes sense to make this as narrow as possible
# maybe not
begin = 380.0 
end = 730.0

# number of wavelengths/channels to solve for
# wavelengths will be non-uniformly spaced
# and will penalize solutions with less than waveVariance
# distance between channels
numwaves = 8
waveVariance = 5.0

# max iterations for solver
maxiter = 20000

# solve additional colors? see munsell.py
# and add your own XYZ colors
solveAdditionalXYZs = False

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
