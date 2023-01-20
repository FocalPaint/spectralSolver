import numpy as np
from settings import *



def spectral_to_RGB(spd, T_MATRIX):
    """Converts n segments spectral power distribution curve to RGB.
    Undoes the offset applies during upsampling
    Based on work by Scott Allen Burns.
    """
    offset = 1.0 - WGM_EPSILON
    r, g, b = np.sum(spd * T_MATRIX, axis=1)
    r = (r - WGM_EPSILON) / offset
    g = (g - WGM_EPSILON) / offset
    b = (b - WGM_EPSILON) / offset
    return r, g, b

def spectral_to_XYZ(spd, T_MATRIX):
    """Converts n segments spectral power distribution curve to XYZ.
    """
    XYZ = np.sum(spd * T_MATRIX, axis=1)
    return XYZ

  
def spectral_Mix_WGM(spd_a, spd_b, ratio):
    """Mixes two SPDs via weighted geomtric mean and returns an SPD.
    Based on work by Scott Allen Burns.
    """
    return np.exp(np.log(spd_a)*(1.0 - ratio) + np.log(spd_b)*ratio)

def rgb_to_Spectral(rgb, spds):
    """Converts RGB to n segments spectral power distribution curve.
    Upsamples to spectral primaries and sums them together into one SPD
    Applies an offset to avoid 0.0
    """
    offset = 1.0 - WGM_EPSILON
    r, g, b = rgb
    r = r * offset + WGM_EPSILON
    g = g * offset + WGM_EPSILON
    b = b * offset + WGM_EPSILON
    # Spectral primaries derived by an optimization routine devised by
    # Allen Burns. Smooth curves <= 1.0 to match XYZ
    
    red = r
    green = g
    blue = b

    whiteSpectrum = np.repeat(1.0, numwaves)
    redSpectrum = spds[0].values
    greenSpectrum = spds[1].values
    blueSpectrum = spds[2].values
    cyanSpectrum = spds[2].values + spds[1].values
    magentaSpectrum = spds[2].values + spds[0].values
    yellowSpectrum = spds[0].values + spds[1].values


    ret = np.repeat(0.0, numwaves)

    # use a technique like Brian Smits uses upsample
    if red <= green and red <= blue:
        ret += red * whiteSpectrum
        if green <= blue:
            ret += (green - red) * cyanSpectrum
            ret += (blue - green) * blueSpectrum
        else:
            ret += (blue - red) * cyanSpectrum
            ret += (green - blue) * greenSpectrum
    elif green <= red and green <= blue:
        ret += green * whiteSpectrum
        if red <= blue:
            ret += (red - green) * magentaSpectrum
            ret += (blue - red) * blueSpectrum
        else:
            ret += (blue - green) * magentaSpectrum
            ret += (red - blue) * redSpectrum
    elif blue <= red and blue <= green:
        ret += blue * whiteSpectrum
        if red <= green:
            ret += (red - blue) * yellowSpectrum
            ret += (green - red) * greenSpectrum
        else:
            ret += (green - blue) * yellowSpectrum
            ret += (red - green) * redSpectrum

    return ret
    # spectral_r = red * spds[0].values

    # spectral_g = green * spds[1].values

    # spectral_b = blue * spds[2].values
    
 
    # return  np.sum([spectral_r, spectral_g, spectral_b], axis=0)


def generateT_MATRIX_RGB(cmfs, illuminant, xyzMatrix):
    cmfs_ = cmfs.transpose()
    T_MATRIX = np.matmul(xyzMatrix, np.matmul(cmfs_, np.diag(illuminant)) # weight for whitepoint
                               / np.matmul(cmfs_[1], illuminant))
    return T_MATRIX


def generateT_MATRIX_XYZ(cmfs, illuminant):
    cmfs_ = cmfs.transpose()
    T_MATRIX = (np.matmul(cmfs_, np.diag(illuminant)) # weight for whitepoint
                               / np.matmul(cmfs_[1], illuminant))
    return T_MATRIX