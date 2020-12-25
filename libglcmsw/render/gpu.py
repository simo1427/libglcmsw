import numpy as np

import concurrent.futures
import multiprocessing
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import itertools
import time

def singletilecpu(im, windowsz, prop,angle, dist,bitdepth, target):
  if target=="cuda":
    from .nvidia import singletilegpusw
    props = ["dissimilarity", "contrast", "homogeneity", "ASM", "energy", "entropy"]
    glcm_hom = singletilegpusw(np.ascontiguousarray(im), windowsz, props.index(prop), dist, angle, bitdepth=bitdepth)
    return glcm_hom

  """elif target=="roc":
    from .roc import singletilegpusw"""


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
  bitdepth = kwargs.get("bitdepth", 256)
  target=kwargs.get("target", "cuda")
  print(f"Using GPU for rendering")
  tiles=[]
  for tile in inptile:
    ni, nj = tile
    tiles.append(img_as_ubyte(rgb2gray(np.load(dpath + f"/{ni}_{nj}.npy"))))
  begintotal = time.perf_counter()
  results = map(singletilecpu, tiles, itertools.repeat(windowsz),
                         itertools.repeat(prop), itertools.repeat(angle), itertools.repeat(distance), itertools.repeat(bitdepth), itertools.repeat(target))
  for p in inptile:
    try:
      ni, nj = p
      np.save(dpath + f"/g{ni}_{nj}.npy", np.ascontiguousarray(next(results)), allow_pickle=True)
      print(p)
    except StopIteration:
      break

  finishtotal = time.perf_counter()
  print(f'Ended in {round(finishtotal - begintotal, 3)}')

