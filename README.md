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
