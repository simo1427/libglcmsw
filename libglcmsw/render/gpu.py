import pyopencl
import os
import numpy as np

import math
import concurrent.futures
import multiprocessing
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import itertools
import time
from . import cpu

def singletilecpu(im, windowsz, prop,angle, dist,bitdepth):
  from numba import cuda

  @cuda.jit("void(float32[:,:], uint8[:,:], uint8, uint8, float32[:])")
  def glcmgen_gpu(glcm, img, x_neighbour, y_neighbour, sum):
    xdims, ydims = img.shape
    xstart = 0
    xend = xdims
    ystart = 0
    yend = ydims
    i, j = cuda.grid(2)
    if x_neighbour < 0:
      xstart += -x_neighbour
    elif x_neighbour >= 0:
      xend = xdims - x_neighbour
    if y_neighbour < 0:
      ystart += -y_neighbour
    elif y_neighbour >= 0:
      yend = ydims - y_neighbour
    if (i >= xstart and i < xend) and ((j >= ystart and j < yend)):
      ref = img[i, j]
      val = img[i + x_neighbour, j + y_neighbour]
      cuda.syncthreads()
      cuda.atomic.add(glcm, (ref, val), 1)
      # glcm[i,j]+=1
      cuda.atomic.add(sum, 0, 1)
      # cuda.syncthreads()
    else:
      return

  #ni,nj=coords
  ri=len(im[:,0])-windowsz+windowsz%2
  rj=len(im[0,:])-windowsz+windowsz%2
  glcm_hom=np.zeros((ri,rj))
  i=0
  j=0
  for ii in range(ri):
    tmp = np.empty((rj,), dtype=np.float32)
    for jj in range(rj):
      img = np.ascontiguousarray(im[ii:ii + windowsz, jj:jj + windowsz])#extract part of the image
      glcm = np.zeros((bitdepth, bitdepth), dtype=np.float64)#calculate glcm
      xdims, ydims = img.shape
      xstart = 0;
      xend = xdims;
      ystart = 0;
      yend = ydims
      x_neighbour = round(dist * np.sin(angle))
      y_neighbour = round(dist * np.cos(angle))
      threadsperblock = (8, 8)
      blockspergrid_x = math.ceil(img.shape[0] / threadsperblock[0])
      blockspergrid_y = math.ceil(img.shape[1] / threadsperblock[1])
      blockspergrid = (blockspergrid_x, blockspergrid_y)
      # print(blockspergrid)
      glcm = np.zeros((256, 256), dtype=np.float32)
      glcm_dev = cuda.to_device(glcm)
      img_dev = cuda.to_device(img)
      sum = np.zeros((1,), dtype=np.float32)
      sum_dev = cuda.to_device(sum)
      glcmgen_gpu[blockspergrid, threadsperblock](glcm_dev, img_dev, x_neighbour, y_neighbour, sum_dev)
      glcm = glcm_dev.copy_to_host()
      glcmtr=np.empty_like(glcm)
      for i in range(bitdepth):
        for j in range(bitdepth):
          glcmtr[i,j]=glcm[j,i]
      glcm = glcm + glcmtr
      div = 0
      for i in range(bitdepth):
        for j in range(bitdepth):
          div+=glcm[i,j]
      for i in range(bitdepth):
        for j in range(bitdepth):
          glcm[i, j]=glcm[i,j]/div
      #####
      #calculate property
      val=cpu.glcmprop(glcm, prop)
      tmp[jj]=val
    glcm_hom[ii]=tmp
    print(ii)

  return np.ascontiguousarray(glcm_hom)

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


def tilerenderlist(dpath, inptile, windowsz, **kwargs):
  workers = kwargs.get("ncores", multiprocessing.cpu_count() // 2 - 1)
  if multiprocessing.cpu_count() < workers or workers < 0:
    raise ValueError("Invalid number of workers")
  prop = kwargs.get("prop", "homogeneity")
  if len(inptile) < workers and len(inptile):
    workers = len(inptile)

  angle = kwargs.get("angle", 0)
  distance = kwargs.get("distance", 1)

  print(f"Using {workers} cores for rendering")
  tiles=[]
  for tile in inptile:
    ni, nj = tile
    tiles.append(img_as_ubyte(rgb2gray(np.load(dpath + f"/{ni}_{nj}.npy"))))
  begintotal = time.perf_counter()
  with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:

    results = executor.map(singletilecpu, tiles, itertools.repeat(windowsz),
                           itertools.repeat(prop), itertools.repeat(angle), itertools.repeat(distance), itertools.repeat(256))
    for p in inptile:
      try:
        ni, nj = p
        np.save(dpath + f"/g{ni}_{nj}.npy", np.ascontiguousarray(next(results)))
        print(p)
      except StopIteration:
        break

  finishtotal = time.perf_counter()
  print(f'Ended in {round(finishtotal - begintotal, 3)}')
