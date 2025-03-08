import matplotlib.pyplot as plt
import colour
import matplotlib.pyplot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from colour.plotting import *
colour_style()

from settings import *
from tools import *



def plotSDS(spds):
    colour.plotting.plot_multi_sds([spds[0], spds[1], spds[2]], use_sds_colours=True, normalise_sds_colours=False)

def draw_primaries(T_MATRIX_XYZ, T_MATRIX_DEVICE):
    colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(colourspaces=colorspace, standalone=False)

    uv_list = []
    # rgb_list = []

    primaries = np.identity(n=numwaves)

    for p in range(numwaves):
        pigment_xyz = spectral_to_XYZ(primaries[p], T_MATRIX_XYZ)
        # rgb_list.append(colour.XYZ_to_RGB(pigment_xyz, colorspacetarget).clip(0,1))
        # rgb_list.append(np.array(spectral_to_RGB(primaries[p], T_MATRIX_DEVICE)).clip(0,1))
        xy = colour.XYZ_to_xy(pigment_xyz)
        uv = colour.xy_to_Luv_uv(xy)
        uv_list.append(uv)
    # whitepoint
    pigment_xyz = spectral_to_XYZ(primaries.sum(axis=1), T_MATRIX_XYZ)
    # rgb_list.append(colour.XYZ_to_RGB(pigment_xyz, colorspacetarget).clip(0,1))
    xy = colour.XYZ_to_xy(pigment_xyz)
    uv = colour.xy_to_Luv_uv(xy)
    uv_list.append(uv)

    matplotlib.pyplot.plot(*zip(*uv_list), 'bo', markersize=5)

    # illuminatnt E
    uv_list = []
    uv_list.append(colour.xy_to_Luv_uv(colour.CCS_ILLUMINANTS['cie_2_1931']['E'].copy()))
    # uv = colour.xy_to_Luv_uv(illuminant_E_xy)
    matplotlib.pyplot.plot(*zip(*uv_list), 'go', markersize=5)

    # plt.figure(figsize = (5,5))
    render(
    standalone=True)

    # plot_RGB_scatter(rgb_list, colorspacetarget, show_spectral_locus=True, points_size=40) 


def draw_colors(color_target, T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs):
    # generate additional columns with 25% and 10% intensity
    colors = colorset #np.concatenate((colorset, colorset *0.18, colorset * 0.09), axis=0)

    # init destination image array
    srgb_colors = np.zeros([51,len(colors) * 3, 3])
    rgb_list = []
    # fill the image with columns of color mixes
    for column, color in enumerate(colors):
        i = 0
        #print ("column", column *3 + 1)
        if column == 0:
            colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(colourspaces=colorspace, standalone=False)
        uv_list = []
        
        for i in range(0, 50):

            ratio = i / 50.
            # mix with linear RGB
            # srgb_colors[i][column*3 + 0] = colour.RGB_to_RGB(np.array(color) * ratio + (1. - ratio) * np.array(color_target), colorspace, colorspaceTargetDevice, apply_cctf_decoding=False,apply_cctf_encoding=True)
            # mix with pigment/spectral upsampling and weighted geometric mean
            pigment_color = spectral_Mix_WGM((rgb_to_Spectral(color_target, primarySDs)), (rgb_to_Spectral(color, primarySDs)), ratio)
            pigment_color_rgb = np.array(spectral_to_RGB(pigment_color, T_MATRIX_DEVICE))
            if column < len(colorset):
                pigment_xyz = spectral_to_XYZ(pigment_color, T_MATRIX_XYZ)
                rgb_list.append(pigment_color_rgb.clip(0,1))
                xy = colour.XYZ_to_xy(pigment_xyz)
                uv = colour.xy_to_Luv_uv(xy)
                uv_list.append(uv)
                
            srgb_colors[i][column*3] = colour.RGB_to_RGB(pigment_color_rgb, colorspaceTargetDevice, colorspaceTargetDevice, apply_cctf_decoding=False, apply_cctf_encoding=True)
            # mix with perceptual RGB (OETF encoded before mixing)
            # srgb_colors[i][column*3 + 2] = colour.RGB_to_RGB(colorspace.cctf_encoding(np.array(color)) * ratio + (1. - ratio) * colorspace.cctf_encoding(np.array(color_target)), colorspace, colorspaceTargetDevice, apply_cctf_decoding=True, apply_cctf_encoding=True)
        
        # matplotlib.pyplot.subplots(subplot_kw={'projection': '3d'})
        
        matplotlib.pyplot.plot(*zip(*uv_list))


        

        
    render(
    standalone=True)

    # print the image and make it bigger
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize = (25,10))
    plt.imshow(srgb_colors)
    render(
    standalone=True)
    plot_RGB_scatter(rgb_list, colour.models.RGB_COLOURSPACE_P3_D65, show_spectral_locus=True, points_size=40) 



def plotColorMixes(T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs):
    print("plot shows pigment mixes only")
    print("\ncolumns order goes linear rgb, spectral weighted geometric mean (pigment), then non-linear rgb (perceptual rgb?)")
    for i in colorset:
        draw_colors(i, T_MATRIX_XYZ, T_MATRIX_DEVICE, primarySDs)