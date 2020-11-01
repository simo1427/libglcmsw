import os
import sys
from skimage import io as si
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import numpy as np
import openslide

"""
func load:
    returns an OpenSlide object - the image file to be processed

    Arguments:
    fpath - path to image file

    Process:
    tries to open file from the following formats:
        Aperio (.svs, .tif)
        Hamamatsu (.vms, .vmu, .ndpi)
        Leica (.scn)
        MIRAX (.mrxs)
        Philips (.tiff)
        Sakura (.svslide)
        Trestile (.tif)
        Ventana (.bif, .tif)
        Generic tiled TIFF (.tif)

    should this be unsuccessful, it tries to open a generic image file
    if nothing works, raises FileNotFoundError
"""
def load(fpath):
    #Loading the image, as well as recovering from any previous crashes
    try:
        openslideobj=openslide.OpenSlide(fpath)
        print("Opened OpenSlide-supported image file")
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        try:
            openslideobj=openslide.open_slide(fpath)
            print("Opened generic image")
        except (openslide.lowlevel.OpenSlideUnsupportedFormatError,FileNotFoundError):
            print("File not found!")
            raise FileNotFoundError
    return openslideobj


"""
func tonpyarr(osobj)
    returns a numpy array, containing the image as a NumPy array

    Arguments:
        osobj - OpenSlide object
        **kwargs:
            downscale - factor by which the image will be downscaled
"""
def tonpyarr(osobj, **kwargs):
    downscalefactor = kwargs.get("downscale", 1)
    w, h = osobj.level_dimensions[0]
    return np.array(osobj.get_thumbnail((round(w/downscalefactor),round(h/downscalefactor))))
