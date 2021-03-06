import os
from skimage import io as si
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
import concurrent.futures
import multiprocessing
import time
from ..io import *
import itertools

"""
func glcmprop:
  returns a float64 - the value of the desired property

  Arguments:
    glcm - a normalized GLCM
    prop - a string containing the desired property
  
  Process:
    if the property is supported by scikit-image, use greycoprops
    elif property is entropy:
      create a copy of the glcm, in which zeros are replaced with ones #avoiding runtimewarnings from numpy
      return the entropy value
  
  Usage:
    render.singletilecpu
    tilegen.reconstruct.fillblanks
"""

def glcmprop(glcm, prop):
  if prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
    return greycoprops(glcm, prop)[0,0]
  elif prop == 'entropy':
    glcm2=glcm
    glcm2[glcm2==0]=1
    return np.sum(glcm[:,:,0,0]*(-np.log(glcm2[:,:,0,0])),axis = None)


"""
func singletilecpu:
  returns a float64 numpy array - result from GLCM Sliding Window

  Arguments:
  coords - tuple containing row and column of the tile
  path - directory containing tile files
  windowsz - size of window for Sliding Window analysis
  prop - property of GLCM to be calculated
  angle - angle of GLCM (0,45,90,...)
  distance - distance of GLCM

  Process:
  parse tile coords;
  read tile image;
  calculate size of returned array (size of tile minus the window size, if windowsz is odd, it increases the dimensions of the resulting image by one)
  iterate through the input image:
    generate a temporary array, which would contain the homogeneity values for the row
    parse a part of the image with the given window size
    generate a normalised GLCM
    calculate homogeneity and store the value in the respective column
    store the temporary array in the respective row of the array with the results

  Usage:
  in libglcmsw.render.cpu.tilerenderlist
"""
def singletilecpu(coords, path, windowsz, prop,angle, distance):
  ni,nj=coords
  im=img_as_ubyte(rgb2gray(np.load(path+f"/{ni}_{nj}.npy")))
  ri=len(im[:,0])-windowsz+windowsz%2
  rj=len(im[0,:])-windowsz+windowsz%2
  glcm_hom=np.zeros((ri,rj))
  i=0
  j=0
  begintotal = time.perf_counter()
  for i in range(ri):
    tmp = np.empty((rj), dtype=np.float32)
    for j in range(rj):
      img = im[i:i + windowsz, j:j + windowsz]
      glcm = greycomatrix(img, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
      tmp[j]=glcmprop(glcm, prop)
    glcm_hom[i]=tmp
    #print(f"Done with {i}")
  finishtotal = time.perf_counter()
  print(f'Processed tile ({ni},{nj}) in {round(finishtotal-begintotal, 3)} seconds')

  return glcm_hom


"""
func singletilecpudebug:
  returns nothing

  Arguments:
  coords - tuple containing row and column of the tile
  path - directory containing tile files
  windowsz - size of window for Sliding Window analysis
  prop - property of GLCM to be calculated

  Process:
  parse tile coords;
  read tile image;
  calculate size of returned array (size of tile minus the window size, if windowsz is odd, it increases the dimensions of the resulting image by one)

  Usage:
  used to debug function libglcmsw.render.tilerenderlist
"""
def singletilecpudebug(coords, path, windowsz, prop):
  begintotal = time.perf_counter()
  ni,nj=coords
  finishtotal = time.perf_counter()
  print(f'Processed tile ({ni},{nj}) in {round(finishtotal-begintotal, 3)} seconds')


"""
func tilerenderlist
  returns nothing

  Arguments:
  dpath - path to tiles directory
  inptile - list of tuples
  windowsz - size of window for sliding window image generation
  **kwargs:
    ncores - number of cores to be used for rendering
    prop - property to be calculated (for GLCM)
    angle - angle of GLCM (0,45,90,...)
    distance - distance of GLCM
    
  Process:
  define number of cores
    if the number of cores is larger than the number of items to be rendered, the workers are reduced to the length of the list
  define property to be calculated
  create a ProcessPoolExecutor:
    get a list for the arguments to be passed to the iterated function (libglcmsw.render.cpu.singletilecpu)
    create a map
    iterate through the list of input tiles (the same size as the number of generators in results!)
    parse coords from tuple
    save the processed image with the prefix 'g' - important for libglcmsw.io.crashrecovery.getunprocessedtiles() and libglcmsw.tiling.reconstruct.*
"""
def tilerenderlist(dpath,inptile,windowsz,**kwargs):
  workers=kwargs.get("ncores",multiprocessing.cpu_count()//2-1)
  if multiprocessing.cpu_count()<workers or workers<0:
    raise ValueError("Invalid number of workers")
  prop=kwargs.get("prop","homogeneity")
  if len(inptile)<workers and len(inptile):
    workers = len(inptile)

  angle=kwargs.get("angle",0)
  distance=kwargs.get("distance",1)

  print(f"Using {workers} cores for rendering")
  begintotal = time.perf_counter()
  with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:

    results = executor.map(singletilecpu, inptile, itertools.repeat(dpath),itertools.repeat(windowsz),itertools.repeat(prop),itertools.repeat(angle),itertools.repeat(distance))
    for p in inptile: 
      try:
        ni, nj=p
        np.save(dpath+f"/g{ni}_{nj}.npy",next(results))
      except StopIteration:
        break
  
  finishtotal = time.perf_counter()
  print(f'Ended in {round(finishtotal-begintotal, 3)}')


"""
func singleline:
  returns a rendered line of the sliding window image (float64)

  Arguments:
  i - row number
  rj - number of columns
  im - image file, from which the segment for the sliding window is appended
  prop - glcm property to be calculated
  PATCH_SIZE - size of sliding window
  angle - angle of GLCM (0,45,90,...)
  distance - distance of GLCM
  
  Process:
  create a new line for the array - tmp
  for every column in the file:
    append the sliding window
    calculate glcm
    calculate property
    add it to the respective element in the tmp array
  return the tmp array
"""
def singleline(i,rj,im,prop,PATCH_SIZE,angle, distance): # processingi of 1 row of the image
  tmp = np.empty((rj), dtype=np.float32)
  for j in range(rj):
    img = im[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
    glcm = greycomatrix(img, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
    tmp[j]=glcmprop(glcm, prop)
  return tmp


"""
func rasterrender:
  returns numpy array - rendered sliding window image (float64)

  Arguments:
  osobj - Openslide object
  windowsz - size of window for sliding window image generation
  **kwargs:
    ncores - number of cores to be used for rendering
    prop - property to be calculated (for GLCM)
    rowssave - number of rows after which a tmp file is saved
    recoveryfile - path to recovery file
    downscale - factor by which the image will be downscaled
    angle - angle of GLCM (0,45,90,...)
    distance - distance of GLCM

  Process:
  define number of coresdefine number of cores
  create a ProcessPoolExecutor:
  define property to be calculated
  create a ProcessPoolExecutor:
    get a list for the arguments to be passed to the iterated function (libglcmsw.render.cpu.singleline)
    create a map
    for every row in the remaining number of rows (after recovery from any crashes)
      save the returned row from the function in the map (libglcmsw.render.cpu.singleline)
    return a numpy array - image as uint8, instead of float64
"""
def rasterrender(osobj,windowsz,**kwargs):
  workers = kwargs.get("ncores", multiprocessing.cpu_count() // 2 - 1)
  if multiprocessing.cpu_count() < workers or workers < 0:
    raise ValueError("Invalid number of workers")
  prop = kwargs.get("prop", "homogeneity")
  ROWSSAVE=kwargs.get("rowssave", 80)
  recoveryfile=kwargs.get("recoveryfile", "./tmpglcmsw.npy")
  dsfact=kwargs.get("downscale", 1)
  PATCH_SIZE=windowsz

  angle = kwargs.get("angle", 0)
  distance = kwargs.get("distance", 1)

  im=img_as_ubyte(rgb2gray(openimg.tonpyarr(osobj,downscale=dsfact)))
  ri=len(im[:,0])-PATCH_SIZE
  rj=len(im[0,:])-PATCH_SIZE
  glcm_hom, k = crashrecovery.rasterrecovery(recoveryfile,ri,rj)
  begintotal = time.perf_counter()
  with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    inp = range(k,ri,1)#TEMP ri=k+ROWSSAVE!!!!
    print(inp)
    results = executor.map(singleline, inp,itertools.repeat(rj), itertools.repeat(im), itertools.repeat(prop), itertools.repeat(windowsz),itertools.repeat(angle), itertools.repeat(distance))
    for p in range(int(k/ROWSSAVE), int(ri/ROWSSAVE+1)): 
      print(p)
      beginlocal=time.perf_counter()
      for i in range(p*ROWSSAVE,(p+1)*ROWSSAVE,1):
        try:
          glcm_hom[i]=next(results)
          print(f'Done with {i}')
        except StopIteration:
          break
      finishlocal=time.perf_counter()
      print(f'Done with {ROWSSAVE} rows in {round(finishlocal-beginlocal, 3)} seconds')
      np.save(recoveryfile, glcm_hom)
  finishtotal = time.perf_counter()
  print(f'Ended in {round(finishtotal-begintotal, 3)}')
  os.remove(recoveryfile)
  return glcm_hom

