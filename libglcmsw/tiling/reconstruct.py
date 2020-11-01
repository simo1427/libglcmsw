from skimage import io as si
from skimage.util import img_as_ubyte
from skimage.feature import greycomatrix,greycoprops
import numpy as np
import time
import os
from . import tilegen
from ..render import cpu


"""
func columnconcat:
    returns a numpy array - concatenation of all processed tiles in a single column

    Arguments:
    dpath - path to directory containing all tiles
    col - number of column (from 0 to ncols-1)
    ncols - number of all rows 

    Process:
    append all images belonging to a single column in a list
    concatenate first 2 tiles in order to initialize numpy array
    iterate through the rest

    Usage:
    in libglcmsw.tiling.allconcat
"""
def columnconcat(dpath,col,ncols):
    im=[]
    for i in range(ncols):
        im.append(np.load(dpath+f"/g{col}_{i}.npy"))
        #print(i,col)
    out=np.concatenate((im[0], im[1]), axis=0)
    for i in range(2,ncols,1):
        out=np.concatenate((out, im[i]),axis=0)
    return out

"""
func allconcat:
    returns the complete image as float64

    Arguments:
    dpath - path to directory containing all tiles
    nrows - number of rows to concatenate
    ncols - number of columns to concatenate

    Process:
    append to a list all reconstructed columns (calling libglcmsw.tiling.reconstruct.columnconcat)
    concateneate first 2 columns to initalize numpy array, containing the result
    iterate through the list, concatenating the rest
    save the file under the name of savefname

    Usage:
    after processing tiles and generating blank tiles
"""
def allconcat(dpath,ncols,nrows):
    img=[]
    for i in range(ncols):
        img.append(columnconcat(dpath,i,nrows))
    out=np.concatenate((img[0], img[1]), axis=1)
    for i in range(2,ncols,1):
        out=np.concatenate((out, img[i]),axis=1)
    return out


"""
func fillblanks:
    returns nothing

    Arguments:
    dpath - path to directory containing tiles
    img - OpenSlide object, source image
    tilesz - Size of a single tile
    ovrlap - size of overlap
    listall - list of all tiles (given by libglcmsw.tiling.tilegen.gettileslistfull)
    **kwargs:
        prop - parse GLCM property, the maximum value of which would be used to fill an empty tile

    Process:
    iterate through all files in the directory:
        if the filename begins with 'g' (meaning that it is processed, check libglcmsw.render.cpu.tilerenderlist)
            parse the is of the tile from the filename
            remove the id of the processed tile
    iterate through the remaining items in the list, which do not have a corresponding rendered tile:
        read the original tile with the same coordinates (check libglcmsw.tiling.tilegen.singletileread)
        save an image with the same naming of a processed tile, completely white, shape similar to the one of the original tile minus the window size and the color dimensions
"""
def fillblanks(dpath,img,tilesz,ovrlap,listall,**kwargs):
    #listall=tilegen.gettileslistfull(img,tilesz,ovrlap)
    #tilegen.tilegendisc(img, tilesz, ovrlap,tmpdir="./deepzoomisch-down30x")
    try:
        prop=kwargs["prop"]
    except KeyError:
        prop="homogeneity"
    angle = kwargs.get("angle", 0)
    distance = kwargs.get("distance", 1)

    tmp = np.full((ovrlap*2+1, ovrlap*2+1),255, dtype=np.uint8)
    glcm = greycomatrix(tmp, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
    fill_value = cpu.glcmprop(glcm, prop)
    for fname in sorted(os.listdir(dpath)):
        if fname[0]=='g':
            tileid=(fname[1:-4]).split("_")
            try:
                listall.remove((int(tileid[0]),int(tileid[1])))
            except:
                pass
    for tileid in listall:
        ni, nj=tileid
        tmp=tilegen.singletileread(img,tilesz,ovrlap,ni,nj)
        #print(tileid)
        #print(tmp.shape)
        #print(tmp[2*ovrlap:,2*ovrlap:,0].shape)
        try:
            np.save(dpath+f"/g{ni}_{nj}.npy", np.full_like(tmp[2*ovrlap:,2*ovrlap:,0],fill_value,dtype=np.float64))
        except:
            pass
