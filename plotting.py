import matplotlib.pyplot as plt
import colour
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from colour.plotting import *
colour_style()

from settings import *
from tools import *



def plotSDS(spds, illuminant_SPD):
    colour.plotting.plot_multi_sds([spds[0], spds[1], spds[2], illuminant_SPD / illuminant_SPD.values.max()], use_sds_colours=True, normalise_sds_colours=False)

def draw_colors(color_target, T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs):
    # generate additional columns with 25% and 10% intensity
    colors = np.concatenate((colorset, colorset *0.18, colorset * 0.09), axis=0)

    # init destination image array
    srgb_colors = np.zeros([51,len(colors) * 3, 3])

    # fill the image with columns of color mixes
    for column, color in enumerate(colors):
        i = 0
        #print ("column", column *3 + 1)
        if column == 0:
            colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(colourspaces=colorspace, standalone=False)
        uv_list = []
        for i in range(0, 51):

            ratio = i / 50.
            # mix with linear RGB
            srgb_colors[i][column*3 + 0] = colour.models.oetf_BT709(np.array(color) * ratio + (1. - ratio) * np.array(color_target))
            # mix with pigment/spectral upsampling and weighted geometric mean
            pigment_color = spectral_Mix_WGM((rgb_to_Spectral(color_target, primarySDs)), (rgb_to_Spectral(color, primarySDs)), ratio)
            pigment_color_rgb = np.array(spectral_to_RGB(pigment_color, T_MATRIX_DEVICE))
            if column < len(colorset):
                pigment_xyz = spectral_to_XYZ(pigment_color, T_MATRIX_XYZ)
                xy = colour.XYZ_to_xy(pigment_xyz)
                uv = colour.xy_to_Luv_uv(xy)
                uv_list.append(uv)
                
            srgb_colors[i][column*3 + 1] = colour.models.oetf_BT709(pigment_color_rgb)
            # mix with perceptual RGB (OETF encoded before mixing)
            srgb_colors[i][column*3 + 2] = colour.models.oetf_BT709(np.array(color)) * ratio + (1. - ratio) * colour.models.oetf_BT709(np.array(color_target))
        
        matplotlib.pyplot.plot(*zip(*uv_list))
        
    render(
    standalone=True)

    # print the image and make it bigger
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize = (25,10))
    plt.imshow(srgb_colors)
    render(
    standalone=True)


def plotColorMixes(T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs):
    print("plot shows pigment mixes only")
    print("\ncolumns order goes linear rgb, spectral weighted geometric mean (pigment), then non-linear rgb (perceptual rgb?)")
    for i in colorset:
        draw_colors(i, T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs)