if __name__ == "__main__":
    import libglcmsw
    from skimage import io as si
    import numpy as np

    #Input data
    img=libglcmsw.io.openimg.load("./input.tif")
    workers=4
    WINDOWSZ=13
    prop="homogeneity"
    output="./outputraster.tif"
    #End of input data

    final=libglcmsw.render.cpu.rasterrender(img,WINDOWSZ,ncores=workers, prop=prop)
    si.imsave(output, final.astype(np.float32))