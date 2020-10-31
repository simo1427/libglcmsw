from . import io
from . import render
from . import tiling
__all__=['io','render','tiling']
import multiprocessing

"""
class SlidingWindowError
defines an error to raise when an error in class SlidingWindow occurs
"""
class SlidingWindowError(Exception):
    pass
"""
class SlidingWindow
wrapper around all code in the example scripts for easy use
Constructor args:
fname - path to image
windowsz - window size,
prop - property
downscale=1 - factor by which the input image is downscaled
"""
class SlidingWindow:
    def __init__(self,fname,windowsz,prop, **kwargs):
        if prop not in["dissimilarity","contrast","homogeneity","ASM","energy","correlation","entropy"]:
            raise SlidingWindowError("Property not available")
        else:
            self.prop=prop
        #####
        downscale=kwargs.get("downscale",1)
        if not downscale==1:
            import openslide
            from PIL import Image
            self.img=openslide.ImageSlide(Image.fromarray(io.openimg.tonpyarr(io.openimg.load(fname),downscale=downscale)))
        else:
            self.img = io.openimg.load(fname)
        #####
        self.windowsz=windowsz
        self.angle=kwargs.get("angle", 0)
        self.distance=kwargs.get("distance",1)
        #####
    """
    method render
    arguments:
    mode (string) - raster or tiling
    tmpdir (string) - path to temporary directory
    workers=cpu_count//2-1
    tilesz=, applicable only for tiling renderer
    rowssave=40, applicable only for raster renderer
    """
    def render(self,mode,tmpdir="./slidingwindowtmp",**kwargs):
        workers=kwargs.get("workers",multiprocessing.cpu_count()//2-1)
        if workers>multiprocessing.cpu_count() or workers<0:
            raise SlidingWindowError("Number of workers does not match the number of available CPU cores")
        if not (mode == "raster" or mode == "tiling"):
            raise SlidingWindowError("Mode not recognized, the available are: 'tiling','raster'")
        elif mode == "raster":
            rowssave=kwargs.get('rowssave', 40)
            self.result = render.cpu.rasterrender(self.img, self.windowsz, ncores=workers, prop=self.prop, rowssave=rowssave, angle=self.angle, distance=self.distance)
        else:
            if not "tilesz" in kwargs:
                raise SlidingWindowError("Tile size must be set for the tiling render mode, otherwise use mode='raster'")
            self.tilesz=kwargs["tilesz"]
            w, h = tiling.tilegen.tilegendisc(self.img, self.tilesz, self.windowsz// 2, tmpdir=tmpdir)
            listoftiles = io.crashrecovery.getunprocessedtiles(tmpdir,tiling.tilegen.gettileslistfull(self.img, self.tilesz, self.windowsz // 2))
            if listoftiles:
                render.cpu.tilerenderlist(tmpdir, listoftiles, self.windowsz, ncores=workers, prop=self.prop, angle=self.angle, distance=self.distance)
                tiling.reconstruct.fillblanks(tmpdir, self.img, self.tilesz, self.windowsz // 2,
                                        tiling.tilegen.gettileslistfull(self.img, self.tilesz,self.windowsz // 2),
                                        prop=self.prop, angle=self.angle, distance=self.distance)
            self.result = tiling.reconstruct.allconcat(tmpdir, w, h)
            from shutil import rmtree
            rmtree(tmpdir)

    """
    method savetif:
    saves self.result in a tif file
    args:
    fname - path to new file without file extension
    type="float32", "float64", "uint8" - data type in the image
    """
    def savetif(self, fname, type="float32"):
        from skimage import io as si
        from skimage.util import img_as_ubyte
        import numpy as np
        if type=="float32":
            si.imsave(fname+".tif",(self.result).astype(np.float32))
        elif type=="float64":
            si.imsave(fname + ".tif", self.result)
        elif type=="uint8":
            si.imsave(fname + ".tif", img_as_ubyte(self.result))
        else:
            raise SlidingWindowError("Unsupported data type. Manually save self.result with any function from image libraries (e.g. scikit-image) and NumPy type conversion")