# Documentation of ```libglcmsw```

This library provides tools for rendering GLCM sliding window images.

## Features

- Raster rendering (recommended for images under 1 MP)
- Tiled rendering (recommended for large images)
- works with biomedical image formats, supported by [openslide](https://github.com/openslide/openslide-python)
- selection of area to be rendered by using a bitmask (BW image with 0 and 255 as the only values)

## Installation

Install openslide binaries (see [Distribution packages, respectively Windows binaries](https://openslide.org/download/))
### Linux

```pip3 install dist/libglcmsw-<version>.whl```

### Windows

```python -m pip install dist/libglcmsw-<version>.whl```

## Rendering modes

There are two ways in which an image could be rendered - using the raster or the tiling renderer.
| Raster | Tiling |
| :---: | :---: |
| under 1 MP images | larger images |
| loads the entire image in memory | separates into smaller images |
| returns a whole image | requires stitching (using `libglcmsw.tiling.reconstruct` submodule) |
| less complicated code | requires more functions to execute correctly |

Keep in mind that the rendering functions create a `ProcessPoolExecutor()`, hence code should be under an `if __name__ == "__main__":` clause.

The name of the properties are analogous to those in `scikit-image`: 
- `contrast`
- `dissimilarity`
- `homogeneity`
- `energy`
- `ASM`
- `correlation`
- `entropy`

### Writing code with the raster renderer

After loading the image with `libglcmsw.io.openimg.load()`, run the function `libglcmsw.render.cpu.rasterrender()` with the required parameters. It returns a numpy array, which could be saved using `skimage.io.imsave()` from the library [`scikit-image`](https://github.com/scikit-image/scikit-image).
An example script can be seen in `examples/rasterrender.py`

### Writing code with the tiling renderer

**Should any of the functions require overlap, it is paramount that it is equal to half of the width of window. If the windows seize is odd, round the result to the lower integer using the `//` operator.**

1. First, separate the image into tiles using `libglcmsw.tiling.tilegen.tilegendisc()`. It returns a tuple - number of tiles along the width and height of the image.
2. create a list of tuples - coordinates of the tiles, which the user wants to render. It can either be a list of the coordinates of all tiles, or a custom selection of such.
3. In order to recover from any previous crashes, use the function `libglcmsw.io.crashrecovery.getunprocessedtiles()` and give as an argument the list mentioned in 2. 
4. Run `libglcmsw.render.cpu.tilerenderlist()` with the desired arguments.
5. If not all tiles have been selected for rendering, use the function `libglcmsw.tiling.reconstruct.fillblanks()` with the needed arguments.
6. To stitch the image, use `libglcmsw.tiling.reconstruct.allconcat()` and in it use the values generated from step 1. It returns a numpy array, which could be saved using `skimage.io.imsave()` from the library [`scikit-image`](https://github.com/scikit-image/scikit-image).

An example script can be seen in `examples/runtilingrenderer.py`.
Example with the **mask selector** is the script `examples/maskselector.py`. It divides both the mask and the image into tiles and checks if any white pixels exist in each tile of the mask. If yes, the respective tile from the original image is included in a list.

## Tree of the module

```
libglcmsw/
    __init__.py
    io/
        __init__.py
        crashrecovery.py
        openimg.py
    render/
        __init__.py
        cpu.py
    tiling/
        __init__.py
        reconstruct.py
        tilegen.py
```
## Index of functions
In this index only functions for use with user code will be described
### io

#### crashrecovery

`libglcmsw.io.crashrecovery.`**`getunprocessedtiles`**`(dpath, tileslist)`:
returns list of tuples - tiles which have not been previously processed for unknown reasons.

Arguments:
1. dpath (string) - path to directory containing all tiles (processed and unprocessed)
2. tileslist - list of tuples - the selection of tiles which are chosen to be processed, given by `libglcmsw.tiling.tilegen.gettileslistdir()` or `libglmcsw.tiling.tilegen.gettileslistfull()`

#### openimg

`libglcmsw.io.openimg.`**`load`**`(fpath)`:
returns an OpenSlide object - the image file to be processed

Arguments:
1. fpath (string) - path to image file

### render

#### cpu

`libglcmsw.render.cpu.`**`tilerenderlist`**`(dpath,inptile,windowsz,**kwargs)`:
returns nothing
Arguments:
1. dpath (string)- path to tiles directory
2. inptile - list of tuples
3. windowsz (int) - size of window for sliding window image generation
4. **kwargs:
    - ncores (int) - number of cores to be used for rendering
    - prop (string)- property to be calculated (for GLCM)

`libglcmsw.render.cpu.`**`rasterrender`**`(osobj,windowsz,**kwargs)`:
returns numpy array - rendered sliding window image (float64)

Arguments:
1. osobj - Openslide object, given by `libglcmsw.io.load()`
2. windowsz (int) - size of window for sliding window image generation
3. **kwargs:
    - ncores (int) - number of cores to be used for rendering
    - prop (string)- property to be calculated (for GLCM)
    - rowssave (int) - number of rows after which a tmp file is saved
    - recoveryfile (string) - path to recovery file - deleted after successful completion
    - downscale (float) - factor by which the image will be downscaled

### tiling

#### reconstruct

`libglcmsw.tiling.reconstruct.`**`allconcat`**`(dpath,ncols,nrows)`:
returns the complete image as float64

Arguments:
1. dpath (string) - path to directory containing all tiles
2. nrows (int) - number of rows to concatenate
3. ncols (int) - number of columns to concatenate

`libglcmsw.tiling.reconstruct.`**`fillblanks`**`(dpath,img,tilesz,ovrlap,listall,**kwargs)`:
returns nothing

Arguments:
1. dpath (string) - path to directory containing all tiles
2. img - OpenSlide object, source image
3. tilesz (int) - Size of a single tile
4. ovrlap (int) - size of overlap
5. listall - list of all tiles (given by libglcmsw.tiling.tilegen.gettileslistfull)
6. **kwargs:
    - prop (string) - parse GLCM property, the maximum value of whichwould be used to fill an empty tile

#### tilegen

`libglcmsw.tiling.tilegen.`**`tilegendisc`**`(openslideobj,tilesz,ovrlap)`:
returns a tuple - the number of useful tiles for rendering
Arguments:
1. demoimg - OpenSlide object
2. tilesz (int) - tile size
3. ovrlap (int) - overlap of tiles
4. **kwargs:
    - tmpdir (string)- temporary directory

`libglcmsw.tiling.tilegen.`**`gettileslistfull`**`(openslideobj,tilesz,ovrlap)`:
return a list of tuples, containing coords of all tiles of an image

Arguments:
1. openslideobj - OpenSlide object

`libglcmsw.tiling.tilegen.`**`gettileslistdir`**`(dpath)`:
return a list of tiles whose originals have been found in the directory containing all itles (i.e. tiles not needed for rendering are deleted)

Arguments:
1. dpath - path to directory containing all tiles
