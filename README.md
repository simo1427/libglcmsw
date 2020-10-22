# libglcmsw

## Introduction
This library provides the necessary tools to generate sliding window images. It accepts a variety of formats: 
- those supported by PIL
- supported by OpenSlide (for biomedical images):
  - Aperio (.svs, .tif)
  - Hamamatsu (.vms, .vmu, .ndpi)
  - Leica (.scn)
  - MIRAX (.mrxs)
  - Philips (.tiff)
  - Sakura (.svslide)
  - Trestile (.tif)
  - Ventana (.bif, .tif)
  - Generic tiled TIFF (.tif) 

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

## Documentation
WIP

## Installation
After installing the openslide binaries (see [Distribution packages, respectively Windows binaries](https://openslide.org/download/)),

```pip3 install dist/libglcmsw-1.0.0-py3-none-any.whl``` (Linux)

```python -m pip install dist/libglcmsw-1.0.0-py3-none-any.whl``` (Windows)

## Note for testing parallel execution

The script is named ```examples/coretest-tilingrenderer.py```. Before running it, under the ```if __name__ == "__main__"``` clause, change the ```coresmax``` variable so that it corresponds to the number of logical processors in the system. The only thing needed is to run the script. It will generate a text file named ```corestest```, to which stdout is redirected. After completion of the testing process, please send this to my email address, along with CPU model and RAM frequency and size (for instance: i7-9750H, DDR4-2666 16GB)