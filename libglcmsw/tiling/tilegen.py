import openslide, openslide.deepzoom
import os, sys
import PIL
import numpy as np
from skimage import io as si
from . import reconstruct

"""
func tilegendisc:
    returns a tuple - the number of useful tiles for rendering

    Arguments:
    demoimg - OpenSlide object
    tilesz - tile size
    ovrlap - overlap of tiles
    **kwargs:
        tmpdir - temporary directory

    Process:
    create a DeepZoomGenerator of the OpenSlide object
    create a temporary directory
    parse the last level of tiles, keeping the full resolution of the image
    check whether the last row and column of the tiles are useful (their dims are larger than the doubled size of the overlap)
    iterate through the number of tiles:
        parse the tile with the coordinates
        save it in png format
"""
def tilegendisc(demoimg,tilesz,ovrlap,**kwargs):
    dz = openslide.deepzoom.DeepZoomGenerator(demoimg, tile_size=tilesz, overlap=ovrlap, limit_bounds=True)
    tmpdir=kwargs.get("tmpdir","./slidingwindowtmp")
    try:
        os.mkdir(tmpdir)
    except FileExistsError:
        while True:
            ans=input("Temporary directory already exists. Do you want to continue? (y/n)")
            if ans=="y":
                break
            elif ans=="n":
                exit()
    os.chdir(tmpdir)
    lc = dz.level_count-1
    ri, rj = dz.level_tiles[lc]#column, row
    lasttile = np.array(dz.get_tile(lc,(ri-1,rj-1)))
    height, width, _ = lasttile.shape
    if width<(2*ovrlap+1):
        ri=ri-1
    if height<(2*ovrlap+1):
        rj = rj-1
    print("Number of tiles",dz.level_tiles[lc])
    for i in range(ri):
        for j in range(rj):
            tmp=dz.get_tile(lc,(i,j))
            #np.save((str(j)+"_"+str(i)+".npy"),np.array(tmp))
            #tmp.save((str(i)+"_"+str(j)+".png"),format="png")#offloading to disc
            np.save(f"{i}_{j}.npy",np.array(tmp))
            #print(f"Done with {(i,j)}")
        #print(f"Done with column {i}")
    os.chdir("..")
    return (ri, rj)
    
"""
func singletileread:
    returns a numpy array of the read tile

    Arguments:
    demoimg - OpenSlide object
    tilesz - tile size
    ovrlap - overlap of tiles
    ni, nj - coordinates of tile

    Process:
    create a DeepZoomGenerator of the OpenSlide object
    parse the last level of tiles, keeping the full resolution of the image
    parse the tile with the coordinates
    return it as a numpy array

    Usage:
    libglcmsw.tiling.reconstruct.fillblanks
"""
def singletileread(demoimg,tilesz,ovrlap,ni,nj):
    dz = openslide.deepzoom.DeepZoomGenerator(demoimg, tile_size=tilesz, overlap=ovrlap, limit_bounds=True)
    lc = dz.level_count-1
    tmp=dz.get_tile(lc,(ni,nj))
    #print(f"Done with tile {ni},{nj}")
    return np.array(tmp)

"""
func gettileslistfull:
    return a list of tuples, containing coords of all tiles of an image

    Arguments:
    openslideobj - OpenSlide object
    tilesz - size of a single tile
    ovrlap - overlap between tiles

    Process:
    create a DeepZoomGenerator of the OpenSlide object
    parse the last level of tiles, keeping the full resolution of the image
    check whether the last row and column of the tiles are useful (their dims are larger than the doubled size of the overlap)
    return a list comprehension with tuples containing coords of all tiles of an image
"""
def gettileslistfull(openslideobj,tilesz,ovrlap):
    dz=openslide.deepzoom.DeepZoomGenerator(openslideobj,tile_size=tilesz,overlap=ovrlap)
    lc = dz.level_count-1
    ri, rj = dz.level_tiles[lc]#column, row
    lasttile = np.array(dz.get_tile(lc,(ri-1,rj-1)))
    height, width, _ = lasttile.shape
    if width<(2*ovrlap+1):
        ri=ri-1
    if height<(2*ovrlap+1):
        rj = rj-1
    print(ri, rj)
    return [(i,j) for i in range(ri) for j in range(rj)]

"""
func gettileslistdir:
    return a list of tiles whose originals have been found in the directory containing all itles (i.e. tiles not needed for rendering are deleted)

    Arguments:
    dpath - path to directory containing all tiles

    Process:
    iterate through the files in the directory
        if the tile is not processed (its name does not begin with 'g'), meaning it it an original tile
            parse the coords from the filename
            add to list a tuple of the coords
    return the list
"""
def gettileslistdir(dpath):
    listoftiles=[]
    for fname in sorted(os.listdir(dpath)):
        #print(fname)
        if not fname[0] == 'g':
            tileid=(fname[0:-4]).split("_")
            listoftiles.append((int(tileid[0]),int(tileid[1])))
    return listoftiles
