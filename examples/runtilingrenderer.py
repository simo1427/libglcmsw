if __name__ == "__main__":
    import libglcmsw
    import openslide
    from skimage import io as si
    import numpy as np
    
    #Input data
    img = libglcmsw.io.openimg.load("./input.tif")
    prop='homogeneity'
    tmpdir="./.tilingtmp"
    WINDOWSZ=13 
    TILESZ=125
    workers=4
    output="./output.tif"
    #End of input data
    
    w, h =libglcmsw.tiling.tilegen.tilegendisc(img,TILESZ,WINDOWSZ//2,tmpdir=tmpdir)
    listoftiles=libglcmsw.io.crashrecovery.getunprocessedtiles(tmpdir,libglcmsw.tiling.tilegen.gettileslistfull(img, TILESZ, WINDOWSZ//2))
    print(listoftiles)
    if not listoftiles:
        print("No tiles to render.")
    else:
        libglcmsw.render.cpu.tilerenderlistnew(tmpdir,listoftiles,WINDOWSZ,ncores=workers, prop=prop)
        libglcmsw.tiling.reconstruct.fillblanks(tmpdir, img, TILESZ, WINDOWSZ // 2,
                                                libglcmsw.tiling.tilegen.gettileslistfull(img, TILESZ, WINDOWSZ // 2),
                                                prop=prop)
    final=libglcmsw.tiling.reconstruct.allconcat(tmpdir,w,h)
    si.imsave(output, final.astype(np.float32))